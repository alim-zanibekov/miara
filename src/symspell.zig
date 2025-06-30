const std = @import("std");
const filter = @import("filter.zig");
const pthash = @import("pthash.zig");
const util = @import("util.zig");
const iterator = @import("iterator.zig");
const bit = @import("bit.zig");
const ef = @import("ef.zig");

/// UTF-8 string type indicator
pub const Utf8 = struct {};

/// SymSpell - wrapper over GenericSymSpell, uses `levenshteinDistance` as a distance function
pub fn SymSpell(
    comptime Word: type,
    comptime Ctx: type,
    comptime wordMaxDistance: fn (ctx: Ctx, str: []const StringElementType(Word)) usize,
) type {
    comptime if (!isStringType(Word)) @compileError("Unsupported string type " ++ @typeName(Word));
    const T = StringElementType(Word);
    const EdT = if (Word == Utf8) u32 else T;

    const Func = struct {
        inline fn editDistanceBuffer(_: Ctx, buffer: []u32, word1: []const EdT, word2: []const EdT) u32 {
            return levenshteinDistance(EdT, u32, buffer, word1, word2);
        }

        inline fn strLen(_: Ctx, word: []const T) ?usize {
            return if (Word == Utf8) std.unicode.utf8CountCodepoints(word) catch null else word.len;
        }
    };

    return GenericSymSpell(Word, Ctx, Func.editDistanceBuffer, Func.strLen, wordMaxDistance, true, true);
}

/// SymSpellDL - wrapper over GenericSymSpell, uses `damerauLevenshteinDistance` as a distance function
pub fn SymSpellDL(
    comptime Word: type,
    comptime Ctx: type,
    comptime wordMaxDistance: fn (ctx: Ctx, str: []const StringElementType(Word)) usize,
) type {
    comptime if (!isStringType(Word)) @compileError("Unsupported string type " ++ @typeName(Word));
    const T = StringElementType(Word);
    const EdT = if (Word == Utf8) u32 else T;

    const Func = struct {
        inline fn editDistanceBuffer(_: Ctx, buffer: []u32, word1: []const EdT, word2: []const EdT) u32 {
            return damerauLevenshteinDistance(EdT, u32, buffer, word1, word2);
        }

        inline fn strLen(_: Ctx, word: []const T) ?usize {
            return if (Word == Utf8) std.unicode.utf8CountCodepoints(word) catch null else word.len;
        }
    };

    return GenericSymSpell(Word, Ctx, Func.editDistanceBuffer, Func.strLen, wordMaxDistance, true, true);
}

/// SymSpell provides a distance function based fuzzy matcher over a dictionary
/// `Word` - slice type of u8, u16, u21, u32 or `Utf8`,
/// `Ctx` - context type for callbacks
/// `wordMaxDistance` - function to get max delete distance for a word
/// `strLen` - function to get string length
/// `compressEdits` - whether to compress the edits index using Elias-Fano compression
/// `bitPackDictRefs` - whether to compress (by bitpacking) the dictionary references
pub fn GenericSymSpell(
    comptime Word: type,
    comptime Ctx: type,
    comptime editDistanceBuffer: edb: {
        const T = StringElementType(Word);
        if (Word == Utf8) {
            break :edb fn (ctx: Ctx, buffer: []u32, word1: []const u32, word2: []const u32) callconv(.@"inline") u32;
        } else {
            break :edb fn (ctx: Ctx, buffer: []u32, word1: []const T, word2: []const T) callconv(.@"inline") u32;
        }
    },
    comptime strLen: fn (ctx: Ctx, str: []const StringElementType(Word)) callconv(.@"inline") ?usize,
    comptime wordMaxDistance: fn (ctx: Ctx, str: []const StringElementType(Word)) usize,
    comptime compressEdits: bool,
    comptime bitPackDictRefs: bool,
) type {
    comptime if (!isStringType(Word)) @compileError("Unsupported string type " ++ @typeName(Word));
    const T = StringElementType(Word);

    const WordEdit = struct { edit: []T, i_word: u32, count: u32 };

    const SortedEditsIterator = struct {
        array: []WordEdit,
        len: usize,
        i: usize = 0,

        pub inline fn size(self: *const @This()) usize {
            return self.len;
        }

        pub fn next(self: *@This()) ?[]const T {
            if (self.i >= self.array.len) return null;
            if (self.i == 0) {
                self.i += 1;
                return self.array[self.i - 1].edit;
            }
            while (std.mem.eql(T, self.array[self.i - 1].edit, self.array[self.i].edit)) {
                self.i += 1;
            }
            self.i += 1;
            return self.array[self.i - 1].edit;
        }
    };

    const PTHash = pthash.PTHash([]const T, pthash.OptimalMapper(u64));
    const EliasFano = ef.EliasFano;

    const debug_stats = builtin.is_test;

    return struct {
        const Self = @This();

        pub const editsCompressed = compressEdits;
        pub const dictRefsBitPacked = bitPackDictRefs;

        pub const Token = struct {
            word: []const T,
            count: u32,
        };

        pub const Hit = struct {
            word: []const T = &.{},
            edit_distance: usize = std.math.maxInt(u32),
            delete_distance: usize = std.math.maxInt(u32),
            count: u32 = 0,
        };

        pub const Suggestion = struct {
            segmented: []T,
            corrected: []T,
            distance_sum: u32,
            probability_log_sum: f64,

            fn init(
                allocator: std.mem.Allocator,
                segmented: []const T,
                corrected: []const T,
                distance_sum: u32,
                probability_log_sum: f64,
            ) !@This() {
                return @This(){
                    .segmented = try allocator.dupe(T, segmented),
                    .corrected = try allocator.dupe(T, corrected),
                    .distance_sum = distance_sum,
                    .probability_log_sum = probability_log_sum,
                };
            }

            fn expand(self: *@This(), allocator: std.mem.Allocator, segmented_len: usize, corrected_len: usize, save_data: bool) !void {
                if (self.segmented.len == 0 and self.corrected.len == 0) {
                    self.segmented = try allocator.alloc(T, segmented_len);
                    self.corrected = try allocator.alloc(T, corrected_len);
                } else {
                    if (save_data) {
                        self.segmented = try allocator.realloc(self.segmented, segmented_len);
                        self.corrected = try allocator.realloc(self.corrected, corrected_len);
                    } else {
                        self.segmented = allocator.remap(self.segmented, segmented_len) orelse sg: {
                            allocator.free(self.segmented);
                            break :sg try allocator.alloc(T, segmented_len);
                        };
                        self.corrected = allocator.remap(self.corrected, corrected_len) orelse sg: {
                            allocator.free(self.corrected);
                            break :sg try allocator.alloc(T, corrected_len);
                        };
                    }
                }
            }

            pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
                allocator.free(self.segmented);
                allocator.free(self.corrected);
            }
        };

        pub const DictStats = struct {
            edits_layer_num_max: usize,
            edits_num_max: usize,
            word_max_len: usize,
            word_max_size: usize,
            max_distance: usize,
            count_sum: usize,
            count_uniform: bool,
        };

        pub const BloomOptions = struct {
            fp_rate: f64, // False positive rate
            dict_stats: ?DictStats = null,
        };

        pub const LRUOptions = struct {
            capacity: u32, // LRU hash map capacity
            dict_stats: ?DictStats = null,
        };

        dict: []const Token,
        dict_stats: DictStats,
        pthash: PTHash.Type,
        edits_index: if (compressEdits) EliasFano else []u32,
        edits_values: if (bitPackDictRefs) bit.BitArray else []u32,
        punctuation: []const []const T = &.{ ",", "]", "[", ")", "(", "}", "{", "." },
        separators: []const []const T = &.{" "},
        segmentation_token: []const T = " ",
        ctx: Ctx,

        assume_ed_increases: bool = true,

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.pthash.deinit(allocator);
            if (compressEdits) {
                self.edits_index.deinit(allocator);
            } else {
                allocator.free(self.edits_index);
            }
            if (bitPackDictRefs) {
                self.edits_values.deinit(allocator);
            } else {
                allocator.free(self.edits_values);
            }
            self.* = undefined;
        }

        /// Collects statistics from the input dictionary.
        /// These stats are used to size buffers, etc
        pub fn getDictStats(ctx: Ctx, dict: []const Token) !DictStats {
            var edits_layer_num_max: usize = 1;
            var edits_num_max: usize = 1;
            var word_max_size: usize = 1;
            var word_max_len: usize = 1;
            var max_distance: usize = 0;
            var count_sum: usize = 0;
            var count_max: u32 = 0;

            var tm = if (debug_stats) util.TimeMasurer{} else (struct {
                pub inline fn start(_: *@This()) void {}
                pub inline fn loop(_: *@This(), _: []const u8) void {}
            }){};

            tm.start();
            for (dict) |it| {
                word_max_size = @max(word_max_size, it.word.len);
                const len = strLen(ctx, it.word) orelse return error.GetWordLengthError;
                word_max_len = @max(word_max_len, len);
                const distance = wordMaxDistance(ctx, it.word);
                max_distance = @max(max_distance, distance);
                const bin = binomial(len, distance);
                edits_layer_num_max = @max(edits_layer_num_max, bin);
                edits_num_max = @max(edits_num_max, maxDeletes(len, distance - 1) + bin);
                count_sum += it.count;
                count_max = @max(it.count, count_max);
            }

            tm.loop("Dict stats calculation");

            return DictStats{
                .edits_layer_num_max = edits_layer_num_max,
                .edits_num_max = edits_num_max,
                .word_max_size = word_max_size,
                .word_max_len = word_max_len,
                .max_distance = max_distance,
                .count_sum = count_sum,
                .count_uniform = (dict.len > 0) and count_max == dict[0].count,
            };
        }

        /// Constructs SymSpell using a Bloom filter for deletes deduplication
        /// `BloomOptions`.`fp_rate` - false positive rate for Bloom filter
        pub fn initBloom(allocator: std.mem.Allocator, dict: []const Token, ctx: Ctx, opts: BloomOptions) !Self {
            const stats = if (opts.dict_stats) |it| it else try getDictStats(ctx, dict);

            const byte_count: usize = @intCast(filter.bloomByteSize(stats.edits_num_max, opts.fp_rate));
            var deduper = try BloomDeduper(T).init(allocator, byte_count, opts.fp_rate);
            defer deduper.deinit();

            return init(allocator, dict, ctx, stats, @TypeOf(&deduper), &deduper);
        }

        /// Constructs SymSpell using an LRU cache for edit deduplication
        /// `LRUOptions`.`capacity` - LRU cache capacity
        pub fn initLRU(allocator: std.mem.Allocator, dict: []const Token, ctx: Ctx, opts: LRUOptions) !Self {
            const stats = if (opts.dict_stats) |it| it else getDictStats(ctx, dict);

            const deduper = try LRUDeduper(T).init(allocator, opts.capacity);
            defer deduper.deinit();

            return init(allocator, dict, ctx, stats, @TypeOf(&deduper), &deduper);
        }

        /// Builds SymSpell dictionary using the provided deduper
        pub fn init(
            main_allocator: std.mem.Allocator,
            dict: []const Token,
            ctx: Ctx,
            dict_stats: DictStats,
            Deduper: type,
            deduper: Deduper,
        ) !Self {
            var tm = if (debug_stats) util.TimeMasurer{} else (struct {
                pub inline fn start(_: *@This()) void {}
                pub inline fn loop(_: *@This(), _: []const u8) void {}
            }){};

            var arena_allocator = std.heap.ArenaAllocator.init(main_allocator);
            defer arena_allocator.deinit();
            const arena = arena_allocator.allocator();

            tm.start();
            var word_edits = try std.ArrayListUnmanaged(WordEdit).initCapacity(arena, dict.len);
            var generator = try EditsGenerator(Word, @TypeOf(deduper)).init(arena, dict_stats.word_max_size);

            for (dict, 0..) |it, i| {
                try generator.load(it.word, wordMaxDistance(ctx, it.word), deduper);
                while (generator.next()) {
                    const str = try arena.dupe(T, generator.getValue());
                    try word_edits.append(arena, WordEdit{
                        .i_word = @intCast(i),
                        .count = it.count,
                        .edit = str,
                    });
                }
            }

            tm.loop("Deletes generation");

            std.mem.sort(WordEdit, word_edits.items, {}, struct {
                fn func(_: void, a: WordEdit, b: WordEdit) bool {
                    const order = std.mem.order(T, a.edit, b.edit);
                    if (order == .eq) return a.i_word < b.i_word;
                    return order == .lt;
                }
            }.func);

            tm.loop("Word deletes sort");

            var unique: usize = 1;
            for (1..word_edits.items.len) |i| {
                if (!std.mem.eql(T, word_edits.items[i].edit, word_edits.items[i - 1].edit)) unique += 1;
            }

            tm.loop("Counting unique deletes");

            var iter = SortedEditsIterator{ .array = word_edits.items, .len = unique };
            var ph = try PTHash.build(main_allocator, @TypeOf(&iter), &iter, PTHash.buildConfig(iter.size(), .{
                .lambda = 6,
                .alpha = 0.97,
                .max_bucket_size = 512,
                // .minimal = true,
            }));

            tm.loop("Build PTHash");

            errdefer ph.deinit(main_allocator);

            var word_edits_groups = try arena.alloc([]WordEdit, ph.size());
            @memset(word_edits_groups, &.{});

            var start_i: usize = 0;
            var prev_idx = try ph.get(word_edits.items[0].edit);
            var edits_values_size: usize = 1;

            // Sort word_edits in EliasFano compressible order, also allows to not have to store counts
            for (1..word_edits.items.len) |i| {
                const hash = try ph.get(word_edits.items[i].edit);
                if (hash == prev_idx) {
                    if (word_edits.items[i].i_word != word_edits.items[i - 1].i_word) edits_values_size += 1;
                    continue;
                }
                edits_values_size += 1;
                word_edits_groups[prev_idx] = word_edits.items[start_i..i];
                start_i = i;
                prev_idx = hash;
            }
            word_edits_groups[prev_idx] = word_edits.items[start_i..];

            tm.loop("Group deletes");

            var edits_index = try (if (compressEdits) arena else main_allocator).alloc(u32, ph.size() + 1);

            const id_width = std.math.log2_int_ceil(usize, dict.len);

            var edits_values = if (bitPackDictRefs)
                try bit.BitArray.initCapacity(main_allocator, id_width * (edits_values_size + 1))
            else
                try main_allocator.alloc(u32, edits_values_size + 1);

            errdefer {
                if (bitPackDictRefs) edits_values.deinit(main_allocator) else main_allocator.free(edits_values);
            }

            @memset(edits_index, 0);
            if (!bitPackDictRefs) @memset(edits_values, 0);

            tm.loop("Prepare buffers");

            var j: u32 = 0;
            var last_hash: u64 = 0;
            for (word_edits_groups, 0..) |group, hash| {
                if (group.len == 0) {
                    // for EliasFano
                    edits_index[hash] = j;
                    continue;
                }
                last_hash = hash;
                edits_index[hash] = j;
                // If we sort edits_values for a group based on count, we can early terminate in Searcher
                // when we already find a word (in edits_values) with the theoretically lowest edit distance for
                // the current delete distance
                std.mem.sort(WordEdit, group, {}, struct {
                    fn func(_: void, a: WordEdit, b: WordEdit) bool {
                        return a.count > b.count;
                    }
                }.func);

                if (comptime bitPackDictRefs) {
                    edits_values.appendUIntAssumeCapacity(group[0].i_word, @intCast(id_width));
                } else {
                    edits_values[j] = group[0].i_word;
                }
                j += 1;
                for (1..group.len) |k| {
                    if (group[k].i_word == group[k - 1].i_word) continue;
                    if (comptime bitPackDictRefs) {
                        edits_values.appendUIntAssumeCapacity(group[k].i_word, @intCast(id_width));
                    } else {
                        edits_values[j] = group[k].i_word;
                    }
                    j += 1;
                }
            }
            // enclosing element to simplify `getEditsIndexAndSize`
            for ((last_hash + 1)..(ph.size() + 1)) |i| edits_index[i] = j;

            tm.loop("Build deletes index");

            if (compressEdits) {
                var ef_iter = (struct {
                    array: []const u32,
                    i: isize = 0,

                    pub fn size(self: *const @This()) usize {
                        return self.array.len;
                    }

                    pub fn next(self: *@This()) ?u64 {
                        if (self.i >= self.array.len) return null;
                        const res = self.array[@intCast(self.i)];
                        self.i += 1;
                        return @intCast(res);
                    }

                    pub fn reset(self: *@This()) void {
                        self.i = 0;
                    }
                }){ .array = edits_index };

                const edits_index_ef = try EliasFano.init(main_allocator, edits_index[edits_index.len - 1], @TypeOf(&ef_iter), &ef_iter);
                // errdefer edits_index_ef.deinit(main_allocator);
                tm.loop("Compress deletes index");
                return Self{
                    .pthash = ph,
                    .edits_values = edits_values,
                    .edits_index = edits_index_ef,
                    .dict = dict,
                    .dict_stats = dict_stats,
                    .ctx = ctx,
                };
            } else {
                return Self{
                    .pthash = ph,
                    .edits_values = edits_values,
                    .edits_index = edits_index,
                    .dict = dict,
                    .dict_stats = dict_stats,
                    .ctx = ctx,
                };
            }
        }

        /// Reusable search instance tied to a specific SymSpell dictionary
        pub fn Searcher(EditsDeduper: type) type {
            return struct {
                sym_spell: *const Self,
                word: []const T,
                seen: std.AutoHashMap(u32, void),
                generator: EditsGenerator(Word, EditsDeduper),
                len: usize,
                max_distance: usize,
                hit: Hit,

                ed_buffer: []u32,
                ed_buffer_1: if (Word == Utf8) []u32 else void,
                ed_buffer_2: if (Word == Utf8) []u32 else void,

                /// Allocates the search buffers and internal state
                pub fn init(sym_spell: *const Self, allocator: std.mem.Allocator) !@This() {
                    const generator = try EditsGenerator(Word, EditsDeduper)
                        .init(
                        allocator,
                        sym_spell.dict_stats.word_max_size + sym_spell.dict_stats.max_distance * 4, // * 4 in case of utf8
                    );
                    var seen = std.AutoHashMap(u32, void).init(allocator);
                    try seen.ensureUnusedCapacity(@min(sym_spell.dict_stats.edits_num_max, 4096)); // magic number, I'm sorry

                    const max_input_len = sym_spell.dict_stats.word_max_len + sym_spell.dict_stats.max_distance;
                    const max_saved_len = sym_spell.dict_stats.word_max_len;

                    return .{
                        .len = undefined,
                        .word = undefined,
                        .max_distance = undefined,
                        .hit = Hit{},
                        .ed_buffer = try allocator.alloc(u32, (max_input_len + 1) * (max_saved_len + 1)),
                        .ed_buffer_1 = if (Word == Utf8) try allocator.alloc(u32, max_saved_len) else undefined,
                        .ed_buffer_2 = if (Word == Utf8) try allocator.alloc(u32, max_input_len) else undefined,
                        .seen = seen,
                        .sym_spell = sym_spell,
                        .generator = generator,
                    };
                }

                pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
                    allocator.free(self.ed_buffer);
                    if (Word == Utf8) {
                        allocator.free(self.ed_buffer_1);
                        allocator.free(self.ed_buffer_2);
                    }
                    self.seen.deinit();
                    self.generator.deinit(allocator);
                }

                /// Initializes the search with a given input word
                pub fn load(self: *@This(), word: []const T, max_distance: usize, edits_deduper: EditsDeduper) !void {
                    const len = strLen(self.sym_spell.ctx, word) orelse return error.GetWordLengthError;
                    if (len > self.sym_spell.dict_stats.word_max_len + self.sym_spell.dict_stats.max_distance) {
                        // no chances to find something useful
                        self.generator.reset();
                        self.len = 0;
                        return;
                    }
                    self.hit = Hit{};
                    self.word = word;
                    self.len = len;
                    self.max_distance = max_distance;
                    self.seen.clearRetainingCapacity();
                    try self.generator.load(word, self.max_distance, edits_deduper);
                }

                fn getEditDistance(self: *@This(), str1: []const T, str2: []const T) !u32 {
                    if (Word == Utf8) {
                        const s1 = try utf8ToU32OnBuffer(self.ed_buffer_1, str1);
                        const s2 = try utf8ToU32OnBuffer(self.ed_buffer_2, str2);
                        return editDistanceBuffer(self.sym_spell.ctx, self.ed_buffer, s1, s2);
                    } else {
                        return editDistanceBuffer(self.sym_spell.ctx, self.ed_buffer, str1, str2);
                    }
                }

                /// Advances the iterator and attempts to find the next closest matching word in the dictionary
                /// Returns true if a new best hit was found
                /// `Searcher`.`hit` is valid until the next call
                pub fn next(self: *@This()) !bool {
                    const id_width = if (bitPackDictRefs) std.math.log2_int_ceil(usize, self.sym_spell.dict.len) else {};

                    var delete_distance: usize = 0;
                    while (self.generator.next()) {
                        const gen_token = self.generator.getValue();
                        const hash = try self.sym_spell.pthash.get(gen_token);
                        const ias = try self.sym_spell.getEditsIndexAndSize(hash);
                        {
                            const word_i = if (bitPackDictRefs)
                                try self.sym_spell.edits_values.getVar(usize, id_width, ias.index * id_width)
                            else
                                self.sym_spell.edits_values[ias.index];

                            const word = self.sym_spell.dict[word_i].word;
                            if (!containsInOrder(T, word, gen_token)) continue;
                        }

                        for ((ias.index + 1)..(ias.index + 1 + ias.size)) |i| {
                            const word_i = if (bitPackDictRefs)
                                try self.sym_spell.edits_values.getVar(usize, id_width, i * id_width)
                            else
                                self.sym_spell.edits_values[i];

                            const it = self.sym_spell.dict[word_i];
                            if (@abs(@as(i64, @intCast(strLen(self.sym_spell.ctx, it.word))) - @as(i64, @intCast(self.len))) > self.max_distance) {
                                continue;
                            }
                            if (self.seen.contains(word_i)) continue;
                            try self.seen.put(word_i, {});

                            const actual_distance = try self.getEditDistance(it.word, gen_token);
                            if (actual_distance <= self.max_distance) {
                                self.hit = .{
                                    .count = it.count,
                                    .word = it.word,
                                    .edit_distance = actual_distance,
                                    .delete_distance = delete_distance,
                                };
                                return true;
                            }
                        }
                        delete_distance = self.generator.current_distance;
                    }
                    return false;
                }

                /// Returns the best match for a term
                pub fn top(self: *@This()) !?Hit {
                    const id_width = if (bitPackDictRefs) std.math.log2_int_ceil(usize, self.sym_spell.dict.len) else {};
                    var best_hit = Hit{};

                    var delete_distance: usize = 0;
                    while (self.generator.next()) {
                        if (self.sym_spell.assume_ed_increases and delete_distance > best_hit.delete_distance) {
                            break;
                        }

                        const gen_token = self.generator.getValue();
                        const hash = try self.sym_spell.pthash.get(gen_token);
                        const ias = try self.sym_spell.getEditsIndexAndSize(hash);
                        if (ias.size == 0) continue;
                        {
                            const word_i = if (bitPackDictRefs)
                                try self.sym_spell.edits_values.getVar(usize, id_width, ias.index * id_width)
                            else
                                self.sym_spell.edits_values[ias.index];
                            const word = self.sym_spell.dict[word_i].word;
                            if (!containsInOrder(T, word, gen_token)) continue;
                        }

                        for ((ias.index)..(ias.index + ias.size)) |i| {
                            const word_i = if (bitPackDictRefs)
                                try self.sym_spell.edits_values.getVar(usize, id_width, i * id_width)
                            else
                                self.sym_spell.edits_values[i];
                            const it = self.sym_spell.dict[word_i];
                            const str_len = strLen(self.sym_spell.ctx, it.word) orelse return error.GetWordLengthError;
                            if (@abs(@as(i64, @intCast(str_len)) - @as(i64, @intCast(self.len))) > self.max_distance) {
                                continue;
                            }

                            // edits_values for a specific word are ordered by count
                            // when best_hit.edit_distance == best_hit.delete_distance we have already found the minimal edit distance
                            if (self.sym_spell.assume_ed_increases and
                                best_hit.edit_distance == best_hit.delete_distance and
                                it.count < best_hit.count) break;

                            if (self.seen.contains(@intCast(word_i))) continue;
                            try self.seen.put(@intCast(word_i), {});

                            const actual_distance = try self.getEditDistance(it.word, gen_token);

                            if (actual_distance <= self.max_distance and
                                (actual_distance < best_hit.edit_distance or
                                    (actual_distance == best_hit.edit_distance and it.count > best_hit.count)))
                            {
                                best_hit = .{
                                    .count = it.count,
                                    .word = it.word,
                                    .edit_distance = actual_distance,
                                    .delete_distance = delete_distance,
                                };

                                if (best_hit.edit_distance == 0) {
                                    return best_hit; // can't do better, exact match
                                }
                            }
                        }
                        delete_distance = self.generator.current_distance;
                    }
                    if (best_hit.word.len > 0) return best_hit;
                    return null;
                }
            };
        }

        pub fn getEditsIndexAndSize(self: *const Self, hash: u64) !struct { index: usize, size: usize } {
            if (compressEdits) {
                const index = try self.edits_index.get(hash);
                const index_next = try self.edits_index.get(hash + 1);
                return .{ .index = index, .size = index_next - index };
            } else {
                const index = self.edits_index[hash];
                const index_next = self.edits_index[hash + 1];
                return .{ .index = index, .size = index_next - index };
            }
        }

        pub fn isStartsWithPunctuation(self: *const Self, str: []const T) ?usize {
            for (self.punctuation) |sp| {
                if (std.mem.startsWith(u8, str, sp)) {
                    return sp.len;
                }
            }
            return null;
        }

        pub fn isStartsWithSeparator(self: *const Self, str: []const T) ?usize {
            for (self.separators) |sp| {
                if (std.mem.startsWith(u8, str, sp)) {
                    return sp.len;
                }
            }
            return null;
        }

        /// Attempts to split and correct a possibly misspelled phrase
        pub fn wordSegmentation(self: *const Self, allocator: std.mem.Allocator, input: []const T) !?Suggestion {
            const input_len = strLen(self.ctx, input) orelse return error.GetWordLengthError;
            const word_max_size = self.dict_stats.word_max_size + self.dict_stats.max_distance * 4;
            const word_max_len = self.dict_stats.word_max_len + self.dict_stats.max_distance;
            const suggestions_count = @min(word_max_len, input_len);

            var suggestions = try allocator.alloc(Suggestion, suggestions_count);
            errdefer allocator.free(suggestions);
            errdefer for (suggestions) |*s| s.deinit(allocator);
            for (suggestions) |*s| s.* = .{ .segmented = &.{}, .corrected = &.{}, .distance_sum = 0, .probability_log_sum = 0.0 };

            var circular_index: isize = -1;

            var searcher = try Self.Searcher(NoopDeduper(u8)).init(self, allocator);
            defer searcher.deinit(allocator);

            var buffer_1 = try allocator.alloc(T, word_max_size);
            defer allocator.free(buffer_1);

            var buffer_2 = try allocator.alloc(T, word_max_size);
            defer allocator.free(buffer_2);

            const space = self.segmentation_token;

            var start: usize = 0;
            for (0..input_len) |j| {
                const max_len = @min(word_max_len, input_len - j);
                var end: usize = start;
                for (1..(max_len + 1)) |i| {
                    if (T == Utf8) {
                        end += try std.unicode.utf8ByteSequenceLength(input[end]);
                    } else {
                        end += 1;
                    }

                    var str = input[start..end];
                    var separator_length: usize = 0;
                    var top_probability_log: f64 = std.math.nan(f64);
                    var top_ed: usize = 0;

                    var k: usize = 0;
                    var str_len: usize = 0;

                    if (self.isStartsWithSeparator(str)) |sp| {
                        str = str[sp..];
                    } else {
                        separator_length += 1;
                    }

                    if (T == Utf8) {
                        var pos: usize = 0;
                        while (pos < str.len) {
                            if (self.isStartsWithSeparator(str[pos..]) orelse self.isStartsWithPunctuation(str[pos..])) |sp| {
                                top_ed += sp;
                                pos += sp;
                                continue;
                            }
                            str_len += 1;
                            const cp_len = std.unicode.utf8ByteSequenceLength(str[pos]) catch unreachable;
                            @memcpy(buffer_1[k..(k + cp_len)], str[pos..(pos + cp_len)]);
                            pos += cp_len;
                            k += cp_len;
                        }
                    } else {
                        var pos: usize = 0;
                        while (pos < str.len) {
                            if (self.isStartsWithSeparator(str[pos..]) orelse self.isStartsWithPunctuation(str[pos..])) |sp| {
                                top_ed += sp;
                                pos += sp;
                                continue;
                            }
                            str_len += 1;
                            buffer_1[k] = str[pos];
                            pos += 1;
                            k += 1;
                        }
                    }

                    const part_length = str_len;
                    const part = buffer_1[0..k];
                    try searcher.load(part, wordMaxDistance(self.ctx, part), NoopDeduper(u8){});

                    var best_hit_ed: usize = std.math.maxInt(usize);
                    var best_hit_dd: usize = std.math.maxInt(usize);
                    var best_hit_word: []const T = &.{};
                    var best_hit_count: u32 = 0;

                    if (try searcher.top()) |top| {
                        const word = top.word;
                        @memcpy(buffer_2[0..word.len], word);
                        best_hit_word = buffer_2[0..word.len];
                        best_hit_ed = top.edit_distance;
                        best_hit_count = top.count;
                        best_hit_dd = top.delete_distance;
                    }

                    var top_result: []const T = undefined;

                    if (best_hit_word.len == 0) {
                        top_ed += part_length;
                        top_result = part;
                        top_probability_log = std.math.log10(10.0 / (@as(f64, @floatFromInt(self.dict_stats.count_sum)) * std.math.pow(f64, @floatFromInt(part_length), 10.0)));
                    } else {
                        top_ed += best_hit_ed;
                        top_result = best_hit_word;
                        top_probability_log = std.math.log10(@as(f64, @floatFromInt(best_hit_count)) / @as(f64, @floatFromInt(self.dict_stats.count_sum)));
                    }

                    if (j == 0) {
                        suggestions[(i - 1) % suggestions_count] = try Suggestion.init(allocator, part, top_result, @intCast(top_ed), top_probability_log);
                    } else {
                        const c: usize = @intCast(circular_index);
                        const d: usize = (i + @as(usize, @intCast(circular_index))) % suggestions_count;

                        if (i == word_max_len or ((suggestions[c].distance_sum + top_ed == suggestions[d].distance_sum or
                            suggestions[c].distance_sum + separator_length + top_ed == suggestions[d].distance_sum) and
                            (suggestions[d].probability_log_sum < suggestions[c].probability_log_sum + top_probability_log)) or
                            suggestions[c].distance_sum + separator_length + top_ed < suggestions[d].distance_sum)
                        {
                            const n_1 = suggestions[c].segmented.len + space.len + part.len;
                            const n_2 = suggestions[c].corrected.len + space.len + top_result.len;

                            if (d == c) {
                                const start_1 = suggestions[d].segmented.len;
                                const start_2 = suggestions[d].corrected.len;
                                try suggestions[d].expand(allocator, n_1, n_2, true);
                                memcpuMultiple(T, suggestions[d].segmented[start_1..], &.{ space, part });
                                memcpuMultiple(T, suggestions[d].corrected[start_2..], &.{ space, top_result });
                            } else {
                                try suggestions[d].expand(allocator, n_1, n_2, false);
                                memcpuMultiple(T, suggestions[d].segmented, &.{ suggestions[c].segmented, space, part });
                                memcpuMultiple(T, suggestions[d].corrected, &.{ suggestions[c].corrected, space, top_result });
                            }

                            suggestions[d].distance_sum = suggestions[c].distance_sum + @as(u32, @intCast(separator_length + top_ed));
                            suggestions[d].probability_log_sum = suggestions[c].probability_log_sum + top_probability_log;
                        }
                    }
                }
                if (T == Utf8) {
                    start += try std.unicode.utf8ByteSequenceLength(input[start]);
                } else {
                    start += 1;
                }
                circular_index += 1;
                if (circular_index >= suggestions_count)
                    circular_index = 0;
            }

            for (suggestions, 0..) |*s, j| {
                if (j != @as(usize, @intCast(circular_index))) s.deinit(allocator);
            }

            if (circular_index != -1) {
                const copy = suggestions[@intCast(circular_index)];
                allocator.free(suggestions);
                return copy;
            } else {
                allocator.free(suggestions);
                return null;
            }
        }
    };
}

/// Does nothing, always allows all edits
pub fn NoopDeduper(comptime T: type) type {
    stringBackingTypeGuard(T);

    return struct {
        const Self = @This();

        fn configure(_: *Self, _: usize, _: usize) void {}
        fn put(_: *Self, _: []const T) void {}
        fn deinit(_: *Self) void {}
        fn check(_: *Self, _: []const T) bool {
            return false;
        }

        pub fn init() Self {
            return Self{};
        }
    };
}

/// Deduper using a hash set to track seen edits
pub fn HashSetDeduper(comptime T: type) type {
    stringBackingTypeGuard(T);

    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        hash_map: *std.StringHashMapUnmanaged(void),
        arena: *std.heap.ArenaAllocator,

        pub fn init(allocator: std.mem.Allocator) !Self {
            const arena = try allocator.create(std.heap.ArenaAllocator);
            arena.child_allocator = allocator;
            arena.state = .{};

            const hash_map = try allocator.create(std.StringHashMapUnmanaged(void));
            hash_map.* = .{};

            return Self{
                .hash_map = hash_map,
                .allocator = allocator,
                .arena = arena,
            };
        }

        fn deinit(self: *Self) void {
            self.arena.deinit();
            self.hash_map.deinit(self.allocator);
            self.allocator.destroy(self.arena);
            self.allocator.destroy(self.hash_map);
            self.* = undefined;
        }

        fn configure(self: *Self, _: usize, _: usize) void {
            _ = self.arena.reset(.retain_capacity);
            self.hash_map.clearRetainingCapacity();
        }

        fn check(self: *Self, word: []const T) bool {
            return self.hash_map.contains(std.mem.sliceAsBytes(word));
        }

        fn put(self: *Self, word: []const T) void {
            const duped_word = self.arena.allocator().dupe(u8, std.mem.sliceAsBytes(word)) catch @panic(@typeName(Self) ++ " out of memory error");
            self.hash_map.put(self.allocator, duped_word, {}) catch @panic(@typeName(Self) ++ " out of memory error");
        }
    };
}

/// Deduper using a Bloom filter to track seen edits
pub fn BloomDeduper(comptime T: type) type {
    stringBackingTypeGuard(T);

    return struct {
        const Self = @This();

        fp_rate: f64,
        buffer: []u8,
        filter: filter.BloomFilter6,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, buffer_size: usize, fp_rate: f64) !Self {
            return Self{
                .filter = undefined,
                .fp_rate = fp_rate,
                .buffer = try allocator.alloc(u8, buffer_size),
                .allocator = allocator,
            };
        }

        fn deinit(self: *Self) void {
            self.allocator.free(self.buffer);
            self.* = undefined;
        }

        fn configure(self: *Self, len: usize, distance: usize) void {
            const n = binomial(len, distance); // worst layer deletes count
            const byte_size = @min(@max(filter.bloomByteSize(n, self.fp_rate), 32), self.buffer.len);
            const n_hash_fn = filter.bloomNHashFn(n, self.fp_rate);

            self.filter = filter.BloomFilter6.initSlice(
                self.buffer[0..byte_size],
                byte_size * 8,
                @min(n_hash_fn, 6),
            ) catch unreachable;

            @memset(self.buffer[0..byte_size], 0);
        }

        fn check(self: *Self, word: []const T) bool {
            return self.filter.contains(std.mem.sliceAsBytes(word));
        }

        fn put(self: *Self, word: []const T) void {
            self.filter.put(std.mem.sliceAsBytes(word));
        }
    };
}

/// Deduper using an LRU cache with fixed capacity
pub fn LRUDeduper(comptime T: type) type {
    stringBackingTypeGuard(T);

    return struct {
        const Self = @This();

        filter: filter.LRUFilter,
        allocator: std.mem.Allocator,

        pub fn init(allocator: std.mem.Allocator, capacity: u32) !Self {
            return Self{
                .filter = try filter.LRUFilter.init(allocator, capacity),
                .allocator = allocator,
            };
        }

        fn deinit(self: *Self) void {
            self.filter.deinit(self.allocator);
            self.* = undefined;
        }

        fn configure(self: *Self, _: usize, _: usize) void {
            self.filter.clear();
        }

        fn check(self: *Self, word: []const T) bool {
            return self.filter.contains(std.mem.sliceAsBytes(word));
        }

        fn put(self: *Self, word: []const T) void {
            self.filter.put(std.mem.sliceAsBytes(word));
        }
    };
}

/// Produces deletion-based edits of a word
/// Uses a deduper to avoid repeating the same edit
pub fn EditsGenerator(comptime TStr: type, TDeduper: type) type {
    comptime if (!isStringType(TStr)) @compileError("Unsupported string type " ++ @typeName(TStr));

    const T = StringElementType(TStr);

    return struct {
        const Self = @This();

        max_distance: usize,
        current_distance: usize,
        current_size: usize,
        buffer: []T,
        word: []const T,
        deletes: []u8,
        deduper: TDeduper,
        word_len: switch (TStr) {
            Utf8 => usize,
            else => void,
        },

        pub fn init(allocator: std.mem.Allocator, buffer_size: usize) !Self {
            const deletes = try allocator.alloc(u8, buffer_size);
            errdefer allocator.free(deletes);

            const buffer = try allocator.alloc(T, buffer_size);
            errdefer allocator.free(buffer);

            return Self{
                .word = &.{},
                .deduper = undefined,
                .max_distance = 0,
                .word_len = switch (TStr) {
                    Utf8 => 0,
                    else => undefined,
                },
                .deletes = deletes,
                .buffer = buffer,
                .current_size = 0,
                .current_distance = 0,
            };
        }

        pub fn load(self: *Self, word: []const T, max_distance: usize, deduper: TDeduper) !void {
            var word_len: usize = undefined;
            if (TStr == Utf8) {
                word_len = try std.unicode.utf8CountCodepoints(word);
                self.word_len = word_len;
            } else {
                word_len = word.len;
            }
            if (word_len > self.deletes.len) return error.Overflow;

            self.word = word;
            self.max_distance = max_distance;
            self.current_distance = 0;
            self.deduper = deduper;

            self.deduper.configure(word_len, max_distance);
            @memset(self.deletes[0..word_len], '_'); // printable, debuggability++
        }

        pub fn reset(self: *Self) void {
            self.word = &.{};
            self.max_distance = 0;
            self.current_distance = 0;
            self.deduper = undefined;
            if (TStr == Utf8) {
                self.word_len = 0;
            }
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.deletes);
            allocator.free(self.buffer);
            self.* = undefined;
        }

        pub fn getValue(self: *const Self) []const T {
            return self.buffer[0..self.current_size];
        }

        pub fn next(self: *Self) bool {
            const n = if (TStr == Utf8) self.word_len else self.word.len;
            if (n == 0) return false;
            const deletes = self.deletes[0..n];
            var buffer: []T = undefined;
            if (TStr == Utf8) {
                var j: usize = 0;
                var i: usize = 0;
                var iter = std.unicode.Utf8Iterator{ .bytes = self.word, .i = 0 };
                while (iter.nextCodepointSlice()) |slice| : (i += 1) {
                    if (deletes[i] == 'x') continue;
                    @memcpy(self.buffer[j..(j + slice.len)], slice);
                    j += slice.len;
                }
                buffer = self.buffer[0..j];
                self.current_size = j;
            } else {
                self.current_size = n - self.current_distance;
                buffer = self.buffer[0..self.current_size];
                var j: usize = 0;
                for (0..n) |i| {
                    if (deletes[i] == 'x') continue;
                    buffer[j] = self.word[i];
                    j += 1;
                }
            }

            if (self.current_distance == 0 or !nextPermutation(u8, deletes)) {
                if (n > self.current_distance and self.current_distance < self.max_distance) {
                    self.current_distance += 1;
                    @memset(deletes, '_');
                    for (deletes[(n - self.current_distance)..]) |*it| it.* = 'x';

                    self.deduper.configure(n, self.max_distance);
                } else {
                    return false;
                }
            }

            if (self.deduper.check(buffer)) {
                return self.next();
            }
            self.deduper.put(buffer);

            return true;
        }
    };
}

/// Calculates Levenshtein distance between two strings
pub fn levenshteinDistance(TStr: type, TBuf: type, dp: []TBuf, a: []const TStr, b: []const TStr) TBuf {
    const m = a.len;
    const n = b.len;

    std.debug.assert(dp.len >= (m + 1) * (n + 1));

    const stride = n + 1;
    const end = m * stride;

    var i: usize = 0;
    var j: usize = 0;
    while (i <= end) : (i += stride) dp[i] = @intCast(i);
    while (j <= n) : (j += 1) dp[j] = @intCast(j);

    i = stride;
    var k: usize = 1;
    while (k <= m) : ({
        k += 1;
        i += stride;
    }) {
        j = 1;
        while (j <= n) : (j += 1) {
            if (a[k - 1] == b[j - 1]) {
                dp[i + j] = dp[i - stride + j - 1];
            } else {
                dp[i + j] = 1 + @min(dp[i - stride + j], @min(dp[i + j - 1], dp[i - stride + j - 1]));
            }
        }
    }

    return dp[m * stride + n];
}

/// Calculates Damerau-Levenshtein distance between two strings (Levenshtein + adjacent swaps)
pub fn damerauLevenshteinDistance(TStr: type, TBuf: type, dp: []TBuf, a: []const TStr, b: []const TStr) TBuf {
    const m = a.len;
    const n = b.len;

    std.debug.assert(dp.len >= (m + 1) * (n + 1));

    const stride = n + 1;
    const end = m * stride;

    var i: usize = 0;
    var j: usize = 0;
    while (i <= end) : (i += stride) dp[i] = @intCast(i / stride);
    while (j <= n) : (j += 1) dp[j] = @intCast(j);

    i = stride;
    var k: usize = 1;
    while (k <= m) : ({
        k += 1;
        i += stride;
    }) {
        j = 1;
        while (j <= n) : (j += 1) {
            if (a[k - 1] == b[j - 1]) {
                dp[i + j] = dp[i - stride + j - 1];
            } else {
                dp[i + j] = 1 + @min(dp[i - stride + j], @min(dp[i + j - 1], dp[i - stride + j - 1]));

                if (k > 1 and j > 1 and a[k - 1] == b[j - 2] and a[k - 2] == b[j - 1]) {
                    dp[i + j] = @min(dp[i + j], dp[i - 2 * stride + j - 2] + 1);
                }
            }
        }
    }

    return dp[m * stride + n];
}

/// Levenshtein distance with buffer allocation
pub fn levenshteinDistanceAlloc(TStr: type, TBuf: type, allocator: std.mem.Allocator, a: []const TStr, b: []const TStr) !TBuf {
    const dp = try allocator.alloc(TBuf, (a.len + 1) * (b.len + 1));
    defer allocator.free(dp);
    return levenshteinDistance(TStr, TBuf, dp, a, b);
}

/// Damerau-Levenshtein distance with buffer allocation
pub fn damerauLevenshteinDistanceAlloc(TStr: type, TBuf: type, allocator: std.mem.Allocator, a: []const TStr, b: []const TStr) !TBuf {
    const dp = try allocator.alloc(TBuf, (a.len + 1) * (b.len + 1));
    defer allocator.free(dp);
    return damerauLevenshteinDistance(TStr, TBuf, dp, a, b);
}

fn stringBackingTypeGuard(comptime T: type) void {
    if (!util.isCharType(T)) @compileError("Unsupported string backing type " ++ @typeName(T));
}

fn isStringType(comptime T: type) bool {
    if (T == Utf8) return true;
    return util.isStringSlice(T);
}

fn containsInOrder(comptime T: type, haystack: []const T, needle: []const T) bool {
    if (needle.len == 0) return true;
    if (haystack.len < needle.len) return false;
    var h: usize = 0;
    var n: usize = 0;
    while (h < haystack.len and n < needle.len) {
        if (std.meta.eql(haystack[h], needle[n])) {
            n += 1;
        }
        h += 1;
    }
    return n == needle.len;
}

fn StringElementType(comptime T: type) type {
    if (T == Utf8) return u8;

    return switch (@typeInfo(T)) {
        .pointer => |ptr| ptr.child,
        .array => |arr| arr.child,
        else => unreachable,
    };
}

test levenshteinDistanceAlloc {
    const allocator = std.testing.allocator;
    const dist = try levenshteinDistanceAlloc(u8, u32, allocator, "unexpectable", "lovely");
    try std.testing.expectEqual(@as(u32, 10), dist);
}

test damerauLevenshteinDistanceAlloc {
    const allocator = std.testing.allocator;
    const dist = try damerauLevenshteinDistanceAlloc(u8, u32, allocator, "bobby", "obbby");
    try std.testing.expectEqual(@as(u32, 1), dist);
}

/// Converts UTF-8 bytes into UTF-32 codepoints on a given buffer
pub fn utf8ToU32OnBuffer(buffer: []u32, utf8: []const u8) error{InvalidUtf8}![]u32 {
    const view = std.unicode.Utf8View.init(utf8) catch return error.InvalidUtf8;
    var it = view.iterator();
    var i: usize = 0;
    while (it.nextCodepoint()) |cp| : (i += 1) {
        buffer[i] = @intCast(cp);
    }

    return buffer[0..i];
}

/// Binomial coefficient nCk
pub fn binomial(n: usize, k: usize) u64 {
    if (k > n) return 0;
    if (k == 0 or k == n) return 1;

    var res: u64 = 1;
    for (0..@min(k, n - k)) |i| {
        res = res * (n - i) / (i + 1);
    }

    return res;
}

/// Maximum number of unique deletes for given word length and delete distance
pub fn maxDeletes(n: usize, d: usize) u64 {
    var sum: u64 = 1;
    for (0..d) |k| {
        sum += binomial(n, k + 1);
    }
    return sum;
}

/// Next lexicographic permutation of the input slice
/// Returns false if the next permutation does not exist
pub fn nextPermutation(T: type, str: []T) bool {
    const len = str.len;
    if (len <= 1) return false;

    var i: usize = len - 2;
    while (true) : (i -= 1) {
        if (str[i] < str[i + 1]) break;
        if (i == 0) return false;
    }

    var j: usize = len - 1;
    while (str[j] <= str[i]) : (j -= 1) {}

    std.mem.swap(T, &str[i], &str[j]);

    std.mem.reverse(T, str[i + 1 ..]);

    return true;
}

fn memcpuMultiple(
    T: type,
    dest: []T,
    sources: []const []const T,
) void {
    var offset: usize = 0;
    for (sources) |src| {
        @memcpy(dest[offset..(offset + src.len)], src);
        offset += src.len;
    }
}
const builtin = @import("builtin");
const testing = std.testing;
const toF64 = util.toF64;

fn testDedupersLaunchCallback(n: u64, testFn: anytype) !void {
    const allocator = testing.allocator;
    std.debug.assert(builtin.is_test);

    const fp_rate = 0.01;
    const bit_count = filter.bloomBitSize(n, fp_rate);
    const buffer_size = std.math.divCeil(usize, bit_count, 8) catch unreachable;

    var bloom_dp = try BloomDeduper(u8).init(allocator, buffer_size, fp_rate);
    defer bloom_dp.deinit();
    try testFn(&bloom_dp, true);

    var lru_dp = try LRUDeduper(u8).init(allocator, @intCast(n / 4));
    defer lru_dp.deinit();
    try testFn(&lru_dp, true);

    var hs_dp = try HashSetDeduper(u8).init(allocator);
    defer hs_dp.deinit();
    try testFn(&hs_dp, true);

    var no_dp = NoopDeduper(u8).init();
    defer no_dp.deinit();
    try testFn(&no_dp, false);
}

test "Edits generation" {
    const allocator = testing.allocator;
    const original_word = "wing";
    const word = "ww" ++ original_word;
    const distance = 2;
    const n = maxDeletes(word.len, distance);

    const testFn = struct {
        fn func(deduper: anytype, chech_dupes: bool) !void {
            std.debug.print("\n---- EditsGeneration | deduper: {s} ----\n\n", .{@typeName(@TypeOf(deduper))});

            std.debug.print("Word:     {s}\n\n", .{word});

            var generator = try EditsGenerator([]const u8, @TypeOf(deduper)).init(allocator, word.len);

            defer generator.deinit(allocator);
            try generator.load(word, distance, deduper);

            var i: usize = 0;
            while (generator.next()) {
                std.debug.print("Edit {d: <4} {s}\n", .{ i, generator.getValue() });
                i += 1;
            }

            if (chech_dupes) {
                try testing.expectEqual(maxDeletes(original_word.len, 2) + (word.len - original_word.len), i);
            }
        }
    }.func;

    try testDedupersLaunchCallback(n, testFn);
}

test "Edits generation | utf8" {
    const allocator = testing.allocator;
    const original_word = "💅💃:hat";
    const word = "💅💅" ++ original_word;
    const distance = 2;
    const word_len = try std.unicode.utf8CountCodepoints(word);
    const n = maxDeletes(word_len, distance);

    const testFn = struct {
        fn func(deduper: anytype, chech_dupes: bool) !void {
            std.debug.print("\n---- EditsGeneration | deduper: {s} ----\n\n", .{@typeName(@TypeOf(deduper))});

            std.debug.print("Word:     {s}\n\n", .{word});

            var generator = try EditsGenerator(Utf8, @TypeOf(deduper))
                .init(allocator, word.len);

            defer generator.deinit(allocator);
            try generator.load(word, distance, deduper);

            var i: usize = 0;
            while (generator.next()) {
                std.debug.print("Edit {d: <4} {s}\n", .{ i, generator.getValue() });
                i += 1;
            }

            if (chech_dupes) {
                const original_word_len = try std.unicode.utf8CountCodepoints(original_word);
                const word_len_l = try std.unicode.utf8CountCodepoints(word);
                try testing.expectEqual(maxDeletes(original_word_len, 2) + (word_len_l - original_word_len), i);
            }
        }
    }.func;

    try testDedupersLaunchCallback(n, testFn);
}

fn testSymSpellRandom(allocator: std.mem.Allocator, SymSpell2: type, n: comptime_int) !void {
    std.debug.assert(builtin.is_test);
    var r = std.Random.DefaultPrng.init(0x4C27A681B1);
    const random = r.random();

    var entries = try allocator.alloc(SymSpell2.Token, n);
    defer {
        for (entries) |s| allocator.free(s.word);
        allocator.free(entries);
    }

    for (0..n) |i| {
        entries[i] = .{
            .word = try util.randomString(allocator, 5 + random.uintLessThan(usize, 10), random),
            .count = 1,
        };
    }

    std.mem.sort(SymSpell2.Token, entries, {}, struct {
        fn func(_: void, a: SymSpell2.Token, b: SymSpell2.Token) bool {
            return std.mem.order(u8, a.word, b.word) == .lt;
        }
    }.func);

    const start = std.time.nanoTimestamp();
    var ss = try SymSpell2.initBloom(allocator, entries, {}, .{ .fp_rate = 0.01 });
    defer ss.deinit(allocator);
    const diff = std.time.nanoTimestamp() - start;

    std.debug.print("Build: ns per key: {d:.2}\n\n", .{toF64(diff) / toF64(entries.len)});

    const query = entries[0].word[0 .. entries[0].word.len - 1];
    var searcher = try SymSpell2.Searcher(NoopDeduper(u8)).init(&ss, allocator);
    defer searcher.deinit(allocator);

    try searcher.load(query, 2, NoopDeduper(u8){});

    const hit = try searcher.top() orelse return error.TestUnexpectedResult;
    try testing.expectEqualSlices(u8, hit.word, entries[0].word);

    const to_segment = try std.mem.concat(allocator, u8, &.{ query, entries[1].word });
    defer allocator.free(to_segment);

    const size_dict = try util.calculateRuntimeSize(allocator, @TypeOf(entries), entries);
    const size = try util.calculateRuntimeSize(allocator, @TypeOf(ss), ss);
    const size_search = try util.calculateRuntimeSize(allocator, @TypeOf(searcher), searcher);
    const size_pthash = try util.calculateRuntimeSize(allocator, @TypeOf(ss.pthash), ss.pthash);
    const size_edits_index = try util.calculateRuntimeSize(allocator, @TypeOf(ss.edits_index), ss.edits_index);
    const size_edits_values = try util.calculateRuntimeSize(allocator, @TypeOf(ss.edits_values), ss.edits_values);

    var hb = util.HumanBytes{};
    std.debug.print(
        "" ++
            "Dict len:         {}\n" ++
            "Deletes count:    {}\n" ++
            "Dict refs count:  {}\n" ++
            "Dict refs size:   {s}\n" ++
            "Edits size:       {s}\n" ++
            "Compress edits:   {}\n" ++
            "Pack dict refs:   {}\n" ++
            "PThash size:      {s}\n" ++
            "Dict size:        {s}\n" ++
            "Searcher size:    {s}\n" ++
            "-----------          \n" ++
            "Total size:       {s}\n",
        .{
            entries.len,
            ss.pthash.size(),
            if (SymSpell2.dictRefsBitPacked)
                (ss.edits_values.len / std.math.log2_int_ceil(usize, entries.len))
            else
                SymSpell2.dictRefsBitPacked,
            hb.fmt(size_edits_values),
            hb.fmt(size_edits_index),
            SymSpell2.editsCompressed,
            SymSpell2.dictRefsBitPacked,
            hb.fmt(size_pthash),
            hb.fmt(size_dict),
            hb.fmt(size_search - size),
            hb.fmt(size),
        },
    );

    std.debug.print("\nWordSegmentation: {s}, original: {s} {s}\n\n", .{ to_segment, entries[0].word, entries[1].word });

    var res = try ss.wordSegmentation(allocator, to_segment) orelse return error.TestUnexpectedResult;
    defer res.deinit(allocator);

    std.debug.print("Segmented: {s}\n", .{res.segmented});
    std.debug.print("Corrected: {s}\n", .{res.corrected});
}

test "SymSpell random ascii" {
    const n = 15000;
    std.debug.print("\n---- SymSpell: random {} ascii words ----\n\n", .{n});
    const SymSpell2 = SymSpell([]const u8, void, struct {
        fn func(_: void, _: []const u8) usize {
            return 2;
        }
    }.func);
    try testSymSpellRandom(testing.allocator, SymSpell2, n);
}

test "SymSpell random ascii uncompressed edits index" {
    const n = 15000;
    std.debug.print("\n---- SymSpell: random {} ascii words, uncompressed edits index ----\n\n", .{n});
    const Func = struct {
        inline fn editDistanceBuffer(_: void, buffer: []u32, word1: []const u8, word2: []const u8) u32 {
            return levenshteinDistance(u8, u32, buffer, word1, word2);
        }
        inline fn strLen(_: void, word: []const u8) ?usize {
            return word.len;
        }
        fn wordMaxDistance(_: void, _: []const u8) usize {
            return 2;
        }
    };
    try testSymSpellRandom(testing.allocator, GenericSymSpell(
        []const u8,
        void,
        Func.editDistanceBuffer,
        Func.strLen,
        Func.wordMaxDistance,
        false,
        false,
    ), n);
}

fn trimSpaces(slice: []const u8) []const u8 {
    return std.mem.trim(u8, slice, " \t");
}

fn getSectionLines(allocator: std.mem.Allocator, content: []const u8, section_name: []const u8) ![][]const u8 {
    var lines = std.ArrayListUnmanaged([]const u8){};
    defer lines.deinit(allocator);

    const start_marker = try std.fmt.allocPrint(allocator, "@{s}", .{section_name});
    defer allocator.free(start_marker);
    const end_marker = try std.fmt.allocPrint(allocator, "@{s}_END", .{section_name});
    defer allocator.free(end_marker);

    var in_section = false;
    var it = std.mem.splitSequence(u8, content, "\n");
    while (it.next()) |line| {
        const trimmed = trimSpaces(line);
        if (std.mem.eql(u8, trimmed, start_marker)) {
            in_section = true;
            continue;
        }
        if (std.mem.eql(u8, trimmed, end_marker)) {
            break;
        }
        if (in_section) {
            const clean_line = if (std.mem.indexOf(u8, trimmed, "#")) |end|
                trimSpaces(trimmed[0..end])
            else
                trimmed;
            if (clean_line.len > 0) {
                try lines.append(allocator, try allocator.dupe(u8, clean_line));
            }
        }
    }

    return try lines.toOwnedSlice(allocator);
}

pub fn testGetSectionLines(section_name: []const u8) ![][]const u8 {
    std.debug.assert(builtin.is_test);

    const allocator = testing.allocator;
    const content = try std.fs.cwd().readFileAlloc(allocator, "./src/symspell_test.txt", std.math.maxInt(usize));
    defer allocator.free(content);

    return try getSectionLines(allocator, content, section_name);
}

fn testSymSpellLoadDict(SymSpellType: type) !struct {
    dict: []const []const u8,
    entries: []SymSpellType.Token,
    sym_spell: SymSpellType,

    fn deinit(self: *@This()) void {
        const allocator = testing.allocator;
        for (self.dict) |it| allocator.free(it);
        allocator.free(self.dict);
        allocator.free(self.entries);
        self.sym_spell.deinit(allocator);
    }
} {
    std.debug.assert(builtin.is_test);

    const allocator = testing.allocator;
    const dict = try testGetSectionLines("DICT");
    errdefer {
        for (dict) |it| allocator.free(it);
        allocator.free(dict);
    }

    var entries = try allocator.alloc(SymSpellType.Token, dict.len);
    errdefer allocator.free(entries);

    for (0..dict.len) |i| {
        var it = std.mem.splitBackwardsSequence(u8, dict[i], " ");
        const count_str = it.next() orelse unreachable;
        const count = try std.fmt.parseInt(u32, count_str, 10);

        entries[i] = .{ .word = dict[i][0..(dict[i].len - count_str.len - 1)], .count = count };
    }

    const sym_spell = try SymSpellType.initBloom(allocator, entries, {}, .{ .fp_rate = 0.01 });
    return .{
        .dict = dict,
        .sym_spell = sym_spell,
        .entries = entries,
    };
}

test "SymSpell fuzzy search" {
    std.debug.print("\n---- SymSpell: fuzzy search ----\n\n", .{});

    const allocator = testing.allocator;
    const SymSpell2 = SymSpellDL(Utf8, void, struct {
        fn func(_: void, _: []const u8) usize {
            return 2;
        }
    }.func);

    var sym_spell_init = try testSymSpellLoadDict(SymSpell2);
    defer sym_spell_init.deinit();
    const sym_spell = sym_spell_init.sym_spell;

    var searcher = try SymSpell2.Searcher(NoopDeduper(u8)).init(&sym_spell, allocator);
    defer searcher.deinit(allocator);

    const to_search = try testGetSectionLines("SEARCH");
    defer {
        for (to_search) |it| allocator.free(it);
        allocator.free(to_search);
    }
    var passed = true;
    for (to_search, 0..) |str, i| {
        var it = std.mem.splitSequence(u8, str, "->");
        const w1 = trimSpaces(it.next() orelse unreachable);
        const w2 = if (it.next()) |s| trimSpaces(s) else null;

        try searcher.load(w1, 2, NoopDeduper(u8){});
        const top = try searcher.top();

        const result = if (top) |hit|
            if (w2 == null)
                testing.expectEqualSlices(u8, &.{}, hit.word)
            else
                testing.expectEqualSlices(u8, w2.?, hit.word)
        else if (w2) |tst|
            testing.expectEqualSlices(u8, tst, &.{})
        else
            unreachable;

        if (result) |_| {
            std.debug.print("Test {d: <4}{s: <30} {s}\n", .{ i, str, "passed" });
        } else |_| {
            std.debug.print("Test {d: <4}{s: <30} {s}\n", .{ i, str, "failed" });
            passed = false;
        }
    }
    try testing.expect(passed);
}

test "SymSpell text segmentation" {
    std.debug.print("\n---- SymSpell: text segmentation ----\n\n", .{});

    const allocator = testing.allocator;
    const SymSpell2 = SymSpellDL(Utf8, void, struct {
        fn func(_: void, _: []const u8) usize {
            return 2;
        }
    }.func);

    var sym_spell_init = try testSymSpellLoadDict(SymSpell2);
    defer sym_spell_init.deinit();
    var sym_spell = sym_spell_init.sym_spell;

    const segmentation = try testGetSectionLines("SEGMENTATION");
    defer {
        for (segmentation) |it| allocator.free(it);
        allocator.free(segmentation);
    }

    var passed = true;
    for (segmentation, 0..) |str, i| {
        var it = std.mem.splitSequence(u8, str, "->");
        const w1 = trimSpaces(it.next() orelse unreachable);
        const w2 = trimSpaces(it.next() orelse unreachable);

        var words_expect = std.mem.splitSequence(u8, w2, "|");
        sym_spell.segmentation_token = "|";
        var segmented = try sym_spell.wordSegmentation(allocator, w1) orelse unreachable;
        defer segmented.deinit(allocator);
        var words = std.mem.splitSequence(u8, segmented.corrected, sym_spell.segmentation_token);

        var success = true;
        while (words_expect.next()) |w| {
            const a = words.next();
            const result = if (a == null)
                testing.expectEqualSlices(u8, w, &.{})
            else
                testing.expectEqualSlices(u8, w, a.?);
            result catch {
                success = false;
                break;
            };
        }

        if (success) {
            std.debug.print("Test {d: <4}{s: <40} {s}\n", .{ i, str, "passed" });
        } else {
            std.debug.print("Test {d: <4}{s: <40} {s}\n", .{ i, str, "failed" });
            passed = false;
        }
    }

    try testing.expect(passed);
}
