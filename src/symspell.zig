const std = @import("std");
const filter = @import("filter.zig");
const pthash = @import("pthash.zig");
const util = @import("util.zig");
const iterator = @import("iterator.zig");
const ef = @import("ef.zig");

const UTF8Buffer = struct {};

fn stringBackingTypeGuard(comptime T: type) void {
    if (!util.isCharType(T)) @compileError("Unsupported string backing type " ++ @typeName(T));
}

fn isStringType(comptime T: type) bool {
    if (T == UTF8Buffer) return true;

    return switch (@typeInfo(T)) {
        .pointer => |ptr| switch (ptr.size) {
            .slice => ptr.child == u8 or ptr.child == u16 or ptr.child == u32 or ptr.child == u21,
            else => false,
        },
        .array => |arr| arr.child == u8 or arr.child == u16 or arr.child == u32 or arr.child == u21,
        else => false,
    };
}

fn StringElementType(comptime T: type) type {
    if (T == UTF8Buffer) return u8;

    return switch (@typeInfo(T)) {
        .pointer => |ptr| ptr.child,
        .array => |arr| arr.child,
        else => unreachable,
    };
}

pub fn SymSpell(
    comptime Word: type,
    comptime compressEdits: bool,
) type {
    comptime if (!isStringType(Word)) @compileError("Unsupported string type " ++ @typeName(Word));
    const T = StringElementType(Word);

    const wordMaxDistance = struct {
        inline fn func(_: []const T) usize {
            return 2;
        }
    }.func;

    const strLen = struct {
        inline fn func(word: []const T) usize {
            return word.len;
        }
    }.func;

    const editDistance = struct {
        inline fn func(allocator: std.mem.Allocator, word1: []const T, word2: []const T) !u32 {
            if (Word == UTF8Buffer) {
                const s1 = try utf8ToU21Slice(allocator, word1);
                defer allocator.free(s1);
                const s2 = try utf8ToU21Slice(allocator, word2);
                defer allocator.free(s2);
                return try idrDistanceAlloc(u21, u32, allocator, s1, s2);
            }
            return try idrDistanceAlloc(T, u32, allocator, word1, word2);
        }
    }.func;

    const editDistanceBuffer = struct {
        inline fn func(buffer: []u32, word1: []const T, word2: []const T) u32 {
            return idrDistance(T, u32, buffer, word1, word2);
        }
    }.func;

    const editDistanceBufferUTF8 = struct {
        inline fn func(buffer: []u32, word1: []const u32, word2: []const u32) u32 {
            return idrDistance(u32, u32, buffer, word1, word2);
        }
    }.func;

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

    const PTHash = pthash.PTHash(Word, pthash.OptimalMapper(u64), SortedEditsIterator);
    const EliasFano = ef.GenericEliasFano(usize);

    return struct {
        const bloom_fp_rate = 0.01;

        const Self = @This();
        pub const Token = struct {
            word: []const T,
            count: u32,
        };

        pub const Hit = struct {
            word: []const T,
            edit_distance: usize,
            delete_distance: usize,
            count: u32,
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

        dict: []Token,
        dict_stats: DictStats,
        pthash: PTHash.Type,
        edits_index_ef: if (compressEdits) EliasFano else void,
        edits_index: if (compressEdits) void else []usize,
        edits_values: []u32,
        punctuation: []const []const T = &.{ &.{','}, &.{']'}, &.{'['}, &.{')'}, &.{'('}, &.{'}'}, &.{'{'}, &.{'.'} },
        separators: []const []const T = &.{&.{' '}},
        segmentation_token: []const T = &.{' '},

        assume_ed_increases: bool = true,

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.pthash.deinit(allocator);
            if (compressEdits) {
                self.edits_index_ef.deinit(allocator);
            } else {
                allocator.free(self.edits_index);
            }
            allocator.free(self.edits_values);
            self.* = undefined;
        }

        pub fn getDictStats(dict: []Token) DictStats {
            var edits_num_max: usize = 1;
            var word_max_size: usize = 1;
            var word_max_len: usize = 1;
            var max_distance: usize = 0;
            var count_sum: usize = 0;
            var count_max: u32 = 0;

            for (dict) |it| {
                word_max_size = @max(word_max_size, it.word.len);
                const len = strLen(it.word);
                word_max_len = @max(word_max_len, len);
                const distance = wordMaxDistance(it.word);
                max_distance = @max(max_distance, distance);
                edits_num_max = @max(edits_num_max, maxDeletes(len, distance));
                count_sum += it.count;
                count_max = @max(it.count, count_max);
            }

            return DictStats{
                .edits_num_max = edits_num_max,
                .word_max_size = word_max_size,
                .word_max_len = word_max_len,
                .max_distance = max_distance,
                .count_sum = count_sum,
                .count_uniform = (dict.len > 0) and count_max == dict[0].count,
            };
        }

        pub fn initBloom(allocator: std.mem.Allocator, dict: []Token, opts: BloomOptions) !Self {
            const stats = if (opts.dict_stats) |it| it else getDictStats(dict);

            const byte_count: usize = @intCast(filter.bloomByteSize(stats.edits_num_max, opts.fp_rate));
            var deduper = try BloomDeduper(T).init(allocator, byte_count, opts.fp_rate);
            defer deduper.deinit();

            return init(allocator, dict, stats, @TypeOf(&deduper), &deduper);
        }

        pub fn initLRU(allocator: std.mem.Allocator, dict: []Token, opts: LRUOptions) !Self {
            const stats = if (opts.dict_stats) |it| it else getDictStats(dict);

            const deduper = try LRUDeduper(T).init(allocator, opts.capacity);
            defer deduper.deinit();

            return init(allocator, dict, stats, @TypeOf(&deduper), &deduper);
        }

        pub fn init(
            main_allocator: std.mem.Allocator,
            dict: []Token,
            dict_stats: DictStats,
            Deduper: type,
            deduper: Deduper,
        ) !Self {
            var arena_allocator = std.heap.ArenaAllocator.init(main_allocator);
            defer arena_allocator.deinit();
            const arena = arena_allocator.allocator();

            var word_edits = try std.ArrayListUnmanaged(WordEdit).initCapacity(arena, dict.len);
            var generator = try EditsGenerator([]const T, @TypeOf(deduper)).init(arena, dict_stats.word_max_size);
            for (dict, 0..) |it, i| {
                try generator.load(it.word, wordMaxDistance(it.word), deduper);
                while (generator.next()) {
                    const str = try arena.dupe(T, generator.getValue());
                    try word_edits.append(arena, WordEdit{
                        .i_word = @intCast(i),
                        .count = it.count,
                        .edit = str,
                    });
                }
            }

            std.mem.sort(WordEdit, word_edits.items, {}, struct {
                fn func(_: void, a: WordEdit, b: WordEdit) bool {
                    const order = std.mem.order(T, a.edit, b.edit);
                    if (order == .eq) return a.i_word < b.i_word;
                    return order == .lt;
                }
            }.func);

            var unique: usize = 1;
            for (1..word_edits.items.len) |i| {
                if (!std.mem.eql(T, word_edits.items[i].edit, word_edits.items[i - 1].edit)) unique += 1;
            }

            var iter = SortedEditsIterator{ .array = word_edits.items, .len = unique };
            var ph = try PTHash.build(main_allocator, &iter, PTHash.buildConfig(iter.size(), .{
                .alpha = 0.97,
                .minimal = true,
            }));
            errdefer ph.deinit(main_allocator);

            if (compressEdits) {
                var word_edits_groups = try arena.alloc([]WordEdit, ph.size());
                @memset(word_edits_groups, &.{});

                var start_i: usize = 0;
                var prev_idx = try ph.get(word_edits.items[0].edit);

                // Sort word_edits in EliasFano compressible order
                for (1..word_edits.items.len) |i| {
                    const hash = try ph.get(word_edits.items[i].edit);
                    if (hash == prev_idx) continue;
                    word_edits_groups[prev_idx] = word_edits.items[start_i..i];
                    start_i = i;
                    prev_idx = hash;
                }
                word_edits_groups[prev_idx] = word_edits.items[start_i..];
                var edits_index = try arena.alloc(usize, ph.size());

                var edits_values = try main_allocator.alloc(u32, word_edits.items.len + ph.size() + 1);
                errdefer main_allocator.free(edits_values);
                @memset(edits_index, 0);
                @memset(edits_values, 0);

                var j: usize = 0;
                for (word_edits_groups, 0..) |group, hash| {
                    if (group.len == 0) {
                        // for EliasFano
                        edits_index[hash] = edits_index[hash -| 1];
                        continue;
                    }

                    edits_index[hash] = j;
                    const i_meta = j;

                    // If we sort edits_values for a group based on count, we can early terminate in Searcher
                    // in case we already find a word (in edits_values) with theoreticaly lowest edit distance for
                    // the current delete distance
                    std.mem.sort(WordEdit, group, {}, struct {
                        fn func(_: void, a: WordEdit, b: WordEdit) bool {
                            return a.count > b.count;
                        }
                    }.func);

                    edits_values[j + 1] = group[0].i_word;
                    j += 2;
                    for (1..group.len) |k| {
                        if (group[k].i_word == group[k - 1].i_word) continue;
                        edits_values[j] = group[k].i_word;
                        j += 1;
                    }
                    edits_values[i_meta] = (@as(u32, @intCast(j - i_meta - 1)) << 16) |
                        (std.hash.Murmur3_32.hash(group[0].edit) & 0x0000FFFF);
                }

                var ef_iter = iterator.SliceIterator(usize).init(edits_index);
                const edits_index_ef = try EliasFano.init(main_allocator, edits_index[edits_index.len - 1], @TypeOf(&ef_iter), &ef_iter);
                // errdefer edits_index_ef.deinit(main_allocator);

                return Self{
                    .pthash = ph,
                    .edits_values = edits_values,
                    .edits_index_ef = edits_index_ef,
                    .dict = dict,
                    .dict_stats = dict_stats,
                    .edits_index = undefined,
                };
            } else {
                var edits_index = try main_allocator.alloc(usize, ph.size());
                errdefer main_allocator.free(edits_index);

                var edits_values = try main_allocator.alloc(u32, word_edits.items.len + ph.size() + 1);
                errdefer main_allocator.free(edits_values);

                @memset(edits_index, 0);
                @memset(edits_values, 0);

                var start_i: usize = 0;
                var prev_idx = try ph.get(word_edits.items[0].edit);
                var j: usize = 0;

                for (1..word_edits.items.len) |i| {
                    const hash = try ph.get(word_edits.items[i].edit);
                    if (hash == prev_idx) continue;

                    edits_index[prev_idx] = j;
                    const i_meta = j;

                    std.mem.sort(WordEdit, word_edits.items[start_i..i], {}, struct {
                        fn func(_: void, a: WordEdit, b: WordEdit) bool {
                            return a.count > b.count;
                        }
                    }.func);

                    edits_values[j + 1] = word_edits.items[start_i].i_word;
                    j += 2;
                    for ((start_i + 1)..i) |k| {
                        if (word_edits.items[k].i_word == word_edits.items[k - 1].i_word) continue;
                        edits_values[j] = word_edits.items[k].i_word;
                        j += 1;
                    }
                    edits_values[i_meta] = (@as(u32, @intCast(j - i_meta - 1)) << 16) |
                        (std.hash.Murmur3_32.hash(word_edits.items[start_i].edit) & 0x0000FFFF);

                    start_i = i;
                    prev_idx = hash;
                }

                return Self{
                    .pthash = ph,
                    .edits_values = edits_values,
                    .edits_index = edits_index,
                    .dict = dict,
                    .dict_stats = dict_stats,
                    .edits_index_ef = undefined,
                };
            }
        }

        pub fn Searcher(EditsDeduper: type) type {
            return struct {
                sym_spell: *const Self,
                word: []const T,
                seen: std.AutoHashMap(u32, void),
                generator: EditsGenerator([]const T, EditsDeduper),
                len: usize,
                max_distance: usize,
                hit: Hit,

                ed_buffer: []u32,
                ed_buffer_1: if (Word == UTF8Buffer) []u32 else void,
                ed_buffer_2: if (Word == UTF8Buffer) []u32 else void,

                pub fn init(sym_spell: *const Self, allocator: std.mem.Allocator) !@This() {
                    const generator = try EditsGenerator([]const T, EditsDeduper)
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
                        .hit = undefined,
                        .ed_buffer = try allocator.alloc(u32, max_input_len * max_saved_len),
                        .ed_buffer_1 = if (Word == UTF8Buffer) try allocator.alloc(u32, max_saved_len) else undefined,
                        .ed_buffer_2 = if (Word == UTF8Buffer) try allocator.alloc(u32, max_input_len) else undefined,
                        .seen = seen,
                        .sym_spell = sym_spell,
                        .generator = generator,
                    };
                }

                pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
                    allocator.free(self.ed_buffer);
                    if (Word == UTF8Buffer) {
                        allocator.free(self.ed_buffer_1);
                        allocator.free(self.ed_buffer_2);
                    }
                    self.seen.deinit();
                    self.generator.deinit(allocator);
                }

                pub fn load(self: *@This(), word: []const T, max_distance: usize, edits_deduper: EditsDeduper) !void {
                    const len = strLen(word);
                    if (len > self.sym_spell.dict_stats.word_max_len + self.sym_spell.dict_stats.max_distance) {
                        // no chances to find something useful
                        self.generator.reset();
                        self.len = 0;
                        return;
                    }
                    self.word = word;
                    self.len = len;
                    self.max_distance = max_distance;
                    self.seen.clearRetainingCapacity();
                    try self.generator.load(word, self.max_distance, edits_deduper);
                }

                fn getEditDistance(self: *@This(), str1: []const T, str2: []const T) u32 {
                    if (Word == UTF8Buffer) {
                        const s1 = try utf8ToU32OnBuffer(self.ed_buffer_1, str1);
                        const s2 = try utf8ToU32OnBuffer(self.ed_buffer_2, str2);
                        return editDistanceBufferUTF8(self.ed_buffer, s1, s2);
                    }
                    return editDistanceBuffer(self.ed_buffer, str1, str2);
                }

                pub fn next(self: *@This()) !bool {
                    while (self.generator.next()) {
                        const gen_token = self.generator.getValue();
                        const hash = try self.sym_spell.pthash.get(gen_token);
                        const d_ref = try self.sym_spell.getEditsIndex(hash);
                        {
                            const mm_hash = std.hash.Murmur3_32.hash(gen_token) & 0x0000FFFF;
                            const mm_hash_chk = self.sym_spell.edits_values[d_ref] & 0x0000FFFF;
                            if (mm_hash != mm_hash_chk) continue;
                        }
                        const size = self.sym_spell.edits_values[d_ref] >> 16;

                        for ((d_ref + 1)..(d_ref + 1 + size)) |i| {
                            const word_i = self.sym_spell.edits_values[i];
                            const it = self.sym_spell.dict[self.sym_spell.edits_values[i]];
                            if (@abs(@as(i64, @intCast(strLen(it.word))) - @as(i64, @intCast(self.len))) > self.max_distance) {
                                continue;
                            }
                            if (self.seen.contains(word_i)) continue;
                            try self.seen.put(word_i, {});

                            const actual_distance = self.getEditDistance(it.word, gen_token);
                            if (actual_distance <= self.max_distance) {
                                self.hit = .{
                                    .count = it.count,
                                    .word = it.word,
                                    .edit_distance = actual_distance,
                                    .delete_distance = self.generator.current_distance,
                                };
                                return true;
                            }
                        }
                    }
                    return false;
                }

                pub fn top(self: *@This()) !?Hit {
                    var best_hit = Hit{
                        .count = 0,
                        .word = &.{},
                        .edit_distance = std.math.maxInt(u32),
                        .delete_distance = std.math.maxInt(u32),
                    };

                    while (self.generator.next()) {
                        const delete_distance = self.generator.current_distance;
                        if (self.sym_spell.assume_ed_increases and delete_distance > best_hit.delete_distance) {
                            break;
                        }

                        const gen_token = self.generator.getValue();
                        const hash = try self.sym_spell.pthash.get(gen_token);
                        const d_ref = try self.sym_spell.getEditsIndex(hash);
                        {
                            const mm_hash = std.hash.Murmur3_32.hash(gen_token) & 0x0000FFFF;
                            const mm_hash_chk = self.sym_spell.edits_values[d_ref] & 0x0000FFFF;
                            if (mm_hash != mm_hash_chk) continue;
                        }
                        const size = self.sym_spell.edits_values[d_ref] >> 16;

                        for ((d_ref + 1)..(d_ref + 1 + size)) |i| {
                            const word_i = self.sym_spell.edits_values[i];
                            const it = self.sym_spell.dict[self.sym_spell.edits_values[i]];
                            if (@abs(@as(i64, @intCast(strLen(it.word))) - @as(i64, @intCast(self.len))) > self.max_distance) {
                                continue;
                            }

                            // edits_values for a specific word are ordered by count
                            // when best_hit.edit_distance == best_hit.delete_distance we already found the minimal edit distance
                            if (self.sym_spell.assume_ed_increases and
                                best_hit.edit_distance == best_hit.delete_distance and
                                it.count < best_hit.count) break;

                            if (self.seen.contains(word_i)) continue;
                            try self.seen.put(word_i, {});

                            const actual_distance = self.getEditDistance(it.word, gen_token);

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
                    }
                    if (best_hit.word.len > 0) return best_hit;
                    return null;
                }
            };
        }

        pub fn getEditsIndex(self: *const Self, hash: u64) !usize {
            if (compressEdits) {
                return try self.edits_index_ef.get(hash);
            } else {
                return self.edits_index[hash];
            }
        }

        pub fn isPunctuation(self: *const Self, str: []const T) bool {
            return findString(T, self.punctuation, str);
        }

        pub fn isSeparator(self: *const Self, str: []const T) bool {
            return findString(T, self.separators, str);
        }

        pub fn search(self: *const Self, allocator: std.mem.Allocator, word: []const T) ![]Hit {
            const len = strLen(word);
            const max_distance = wordMaxDistance(word);

            const n = maxDeletes(len, max_distance);
            const bit_count = filter.bloomBitSize(n, bloom_fp_rate);
            const buffer_size = std.math.divCeil(usize, bit_count, 8) catch unreachable;
            var deduper = try BloomDeduper(u8).init(allocator, buffer_size, bloom_fp_rate);
            defer deduper.deinit();
            var generator = try EditsGenerator(Word, @TypeOf(deduper)).init(allocator, word.len);
            defer generator.deinit(allocator);
            try generator.load(word, max_distance, deduper);

            var result = std.ArrayListUnmanaged(Hit){};
            errdefer result.deinit(allocator);

            var seen = std.AutoHashMapUnmanaged(u32, void){};
            defer seen.deinit(allocator);

            while (generator.next()) {
                const gen_token = generator.getValue();
                const hash = try self.pthash.get(gen_token);
                const d_ref = try self.getEditsIndex(hash);
                {
                    const mm_hash = std.hash.Murmur3_32.hash(generator.getValue()) & 0x0000FFFF;
                    const mm_hash_chk = self.edits_values[d_ref] & 0x0000FFFF;
                    if (mm_hash != mm_hash_chk) continue;
                }
                const size = self.edits_values[d_ref] >> 16;

                for ((d_ref + 1)..(d_ref + 1 + size)) |i| {
                    const word_i = self.edits_values[i];
                    const it = self.dict[self.edits_values[i]];
                    if (@abs(@as(i64, @intCast(strLen(it.word))) - @as(i64, @intCast(len))) > max_distance) {
                        continue;
                    }

                    if (seen.contains(word_i)) continue;
                    try seen.put(allocator, word_i, {});

                    const actual_distance = try editDistance(allocator, it.word, gen_token);
                    if (actual_distance <= max_distance) {
                        try result.append(allocator, .{
                            .count = it.count,
                            .word = it.word,
                            .edit_distance = actual_distance,
                            .delete_distance = generator.current_distance,
                        });
                    }
                }
            }

            return result.toOwnedSlice(allocator);
        }

        pub fn wordSegmentation(self: *const Self, allocator: std.mem.Allocator, input: []const T) !?Suggestion {
            const input_len = strLen(input);
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
                    end += try std.unicode.utf8ByteSequenceLength(input[end]);
                    var str = input[start..end];
                    var separator_length: usize = 0;
                    var top_probability_log: f64 = std.math.nan(f64);
                    var top_ed: usize = 0;

                    const first_len = try std.unicode.utf8ByteSequenceLength(str[0]);
                    if (self.isSeparator(str[0..first_len])) {
                        str = str[first_len..];
                    } else {
                        separator_length += 1;
                    }

                    const view = std.unicode.Utf8View.initUnchecked(str);
                    var it = view.iterator();

                    var k: usize = 0;
                    var str_len: usize = 0;

                    while (it.nextCodepointSlice()) |cp| {
                        str_len += 1;
                        if (self.isSeparator(cp) or self.isPunctuation(cp)) {
                            top_ed += 1;
                            continue;
                        }
                        @memcpy(buffer_1[k..(k + cp.len)], cp);
                        k += cp.len;
                    }

                    const part_length = str_len;
                    const part = buffer_1[0..k];
                    try searcher.load(part, wordMaxDistance(part), NoopDeduper(u8){});

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
                                const start_2 = suggestions[d].segmented.len;
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

                start += try std.unicode.utf8ByteSequenceLength(input[start]);
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

fn NoopDeduper(comptime T: type) type {
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

fn HashSetDeduper(comptime T: type) type {
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

fn BloomDeduper(comptime T: type) type {
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
            const n = maxDeletes(len, distance);
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

fn LRUDeduper(comptime T: type) type {
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

fn EditsGenerator(comptime TStr: type, TDeduper: type) type {
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
            UTF8Buffer => usize,
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
                    UTF8Buffer => 0,
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
            if (TStr == UTF8Buffer) {
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
            if (TStr == UTF8Buffer) {
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
            const n = if (TStr == UTF8Buffer) self.word_len else self.word.len;
            if (n == 0) return false;
            const deletes = self.deletes[0..n];
            var buffer: []T = undefined;
            if (TStr == UTF8Buffer) {
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

            // std.debug.print("{s} {s}\n", .{ buffer, deletes });

            if (self.current_distance == 0 or !nextPermutation(u8, deletes)) {
                if (n > self.current_distance and self.current_distance < self.max_distance) {
                    self.current_distance += 1;
                    @memset(deletes, '_');
                    for (deletes[(n - self.current_distance)..]) |*it| it.* = 'x';
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

fn findString(T: type, haystack: []const []const T, needle: []const T) bool {
    for (haystack) |str| {
        if (std.mem.eql(u8, str, needle)) return true;
    }
    return false;
}

pub fn idrDistance(T: type, Buf: type, dp: []Buf, a: []const T, b: []const T) Buf {
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

pub fn idrDistanceAlloc(TStr: type, TBuf: type, allocator: std.mem.Allocator, a: []const TStr, b: []const TStr) !TBuf {
    const dp = try allocator.alloc(TBuf, (a.len + 1) * (b.len + 1));
    defer allocator.free(dp);
    return idrDistance(TStr, TBuf, dp, a, b);
}

test idrDistanceAlloc {
    const allocator = std.testing.allocator;
    const dist = try idrDistanceAlloc(u8, u32, allocator, "unexpectable", "lovely");
    try std.testing.expectEqual(@as(u32, 10), dist);
}

pub fn utf8ToU21Slice(allocator: std.mem.Allocator, utf8: []const u8) error{InvalidUtf8}![]u21 {
    var it = std.unicode.Utf8View.init(utf8) catch return error.InvalidUtf8;
    var codepoints = std.ArrayList(u21).init(allocator);

    while (it.nextCodepoint()) |cp| {
        try codepoints.append(cp);
    } else |err| {
        return err;
    }

    return codepoints.toOwnedSlice();
}

pub fn utf8ToU32OnBuffer(buffer: []u32, utf8: []const u8) error{InvalidUtf8}![]u32 {
    var it = std.unicode.Utf8View.init(utf8) catch return error.InvalidUtf8;

    var i: usize = 0;
    while (it.nextCodepoint()) |cp| : (i += 1) {
        buffer[i] = @intCast(cp);
    } else |err| {
        return err;
    }

    return buffer[0..i];
}

pub fn binomial(n: usize, k: usize) u64 {
    if (k > n) return 0;
    if (k == 0 or k == n) return 1;

    var res: u64 = 1;
    for (0..@min(k, n - k)) |i| {
        res = res * (n - i) / (i + 1);
    }

    return res;
}

pub fn maxDeletes(n: usize, d: usize) u64 {
    var sum: u64 = 0;
    for (0..d) |k| {
        sum += binomial(n, k + 1);
    }
    return sum;
}

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

fn runTestFn(n: u64, allocator: std.mem.Allocator, testFn: anytype) !void {
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

test "EditsGeneration" {
    const allocator = std.testing.allocator;
    const original_word = "wing";
    const word = "ww" ++ original_word;
    const distance = 2;
    const n = maxDeletes(word.len, distance);

    const testFn = struct {
        fn func(deduper: anytype, chech_dupes: bool) !void {
            std.debug.print("\n---- EditsGeneration | deduper: {s} ----\n\n", .{@typeName(@TypeOf(deduper))});

            std.debug.print("Word: {s}\n", .{word});

            var generator = try EditsGenerator([]const u8, @TypeOf(deduper)).init(allocator, word.len);

            defer generator.deinit(allocator);
            try generator.load(word, distance, deduper);

            var i: usize = 0;
            while (generator.next()) {
                std.debug.print("Edit[{}]:\t{s}\n", .{ i, generator.getValue() });
                i += 1;
            }

            if (chech_dupes) {
                try std.testing.expectEqual(maxDeletes(original_word.len, 2) + (word.len - original_word.len), i);
            }
        }
    }.func;

    try runTestFn(n, allocator, testFn);
}

test "EditsGenerationUTF8" {
    const allocator = std.testing.allocator;
    const original_word = "💅💃:hat";
    const word = "💅💅" ++ original_word;
    const distance = 2;
    const word_len = try std.unicode.utf8CountCodepoints(word);
    const n = maxDeletes(word_len, distance);

    const testFn = struct {
        fn func(deduper: anytype, chech_dupes: bool) !void {
            std.debug.print("\n---- EditsGeneration | deduper: {s} ----\n\n", .{@typeName(@TypeOf(deduper))});

            std.debug.print("Word: {s}\n", .{word});

            var generator = try EditsGenerator(UTF8Buffer, @TypeOf(deduper))
                .init(allocator, word.len);

            defer generator.deinit(allocator);
            try generator.load(word, distance, deduper);

            var i: usize = 0;
            while (generator.next()) {
                std.debug.print("Edit[{}]:\t{s}\n", .{ i, generator.getValue() });
                i += 1;
            }

            if (chech_dupes) {
                const original_word_len = try std.unicode.utf8CountCodepoints(original_word);
                const word_len_l = try std.unicode.utf8CountCodepoints(word);
                try std.testing.expectEqual(maxDeletes(original_word_len, 2) + (word_len_l - original_word_len), i);
            }
        }
    }.func;

    try runTestFn(n, allocator, testFn);
}

test "SymSpell" {
    const allocator = std.testing.allocator;
    const n = 500;
    var r = std.Random.DefaultPrng.init(0x4C27A681B1);
    const random = r.random();

    const Type = SymSpell([]const u8, true);

    var entries = try allocator.alloc(Type.Token, n);
    defer {
        for (entries) |s| allocator.free(s.word);
        allocator.free(entries);
    }

    for (0..n) |i| {
        entries[i] = .{
            .word = try util.randomString(allocator, 5 + random.uintLessThan(usize, 20), random),
            .count = 1,
        };
    }

    var ss = try Type.initBloom(allocator, entries, .{ .fp_rate = 0.01 });
    defer ss.deinit(allocator);

    const query = entries[0].word[0 .. entries[0].word.len - 1];
    const hits = try ss.search(allocator, query);
    defer allocator.free(hits);

    std.debug.print("\n---- SymSpell - Search | method default, query: {s} ----\n\n", .{query});

    for (hits, 0..) |hit, i| {
        std.debug.print("Hit[{}]: {s}, {}\n", .{ i, hit.word, hit.edit_distance });
    }

    std.debug.print("\n---- SymSpell - Search | method prealloc, query: {s} ----\n\n", .{query});

    var searcher = try Type.Searcher(NoopDeduper(u8)).init(&ss, allocator);
    defer searcher.deinit(allocator);

    // we can use this searcher multiple times, and it will not alloc (in the vast majority of cases)
    // the exceptions are when we search for a word that has more common deletes with words than any other word in the dict
    try searcher.load(query, 2, NoopDeduper(u8){});
    var i: usize = 0;
    while (try searcher.next()) : (i += 1) {
        std.debug.print("Hit[{}]: {s}, {}\n", .{ i, searcher.hit.word, searcher.hit.edit_distance });
    }

    var res = try ss.wordSegmentation(allocator, "oF6Mgg3isR0HoF6Mgg3isR0H");
    std.debug.print("Segmented {s}\n", .{res.?.segmented});
    std.debug.print("Corrected {s}\n", .{res.?.corrected});
    if (res != null) res.?.deinit(allocator);

    try std.testing.expect(hits.len > 0);
    try std.testing.expectEqualStrings(hits[0].word, entries[0].word);
}
