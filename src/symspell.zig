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
    return util.isStringSlice(T);
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
) type {
    comptime if (!isStringType(Word)) @compileError("Unsupported string type " ++ @typeName(Word));
    const T = StringElementType(Word);

    const wordMaxDistance = struct {
        inline fn func(_: []const T) usize {
            return 2;
        }
    }.func;

    const wordLen = struct {
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

    const WordEdit = struct { i_word: u32, edit: []T };

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
            count: u32,
        };

        pub const Suggestion = struct {
            segmented: []T,
            corrected: []T,
            distance_sum: u32,
            probability_log_sum: f64,
        };

        pub const DictStats = struct {
            edits_num_max: usize,
            word_max_size: usize,
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
        pthash: PTHash.Type,
        edits_index: EliasFano,
        edits_values: []u32,

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.pthash.deinit(allocator);
            self.edits_index.deinit(allocator);
            // allocator.free(self.edits_index);
            allocator.free(self.edits_values);
            self.* = undefined;
        }

        pub fn getDictStats(dict: []Token) DictStats {
            var edits_num_max: usize = 1;
            var word_max_size: usize = 1;

            for (dict) |it| {
                word_max_size = @max(word_max_size, it.word.len);
                const len = wordLen(it.word);
                edits_num_max = @max(edits_num_max, maxDeletes(len, wordMaxDistance(it.word)));
            }

            return DictStats{ .edits_num_max = edits_num_max, .word_max_size = word_max_size };
        }

        pub fn initBloom(allocator: std.mem.Allocator, dict: []Token, opts: BloomOptions) !Self {
            const stats = if (opts.dict_stats) |it| it else getDictStats(dict);

            const byte_count: usize = @intCast(filter.bloomByteSize(stats.edits_num_max, opts.fp_rate));
            var deduper = try BloomDeduper(T).init(allocator, byte_count, opts.fp_rate);
            defer deduper.deinit();

            return init(allocator, dict, stats.edits_num_max, stats.word_max_size, &deduper);
        }

        pub fn initLRU(allocator: std.mem.Allocator, dict: []Token, opts: LRUOptions) !Self {
            const stats = if (opts.dict_stats) |it| it else getDictStats(dict);

            const deduper = try LRUDeduper(T).init(allocator, opts.capacity);
            defer deduper.deinit();

            return init(allocator, dict, stats.edits_num_max, stats.word_max_size, &deduper);
        }

        pub fn init(
            main_allocator: std.mem.Allocator,
            dict: []Token,
            _: usize, // edits_num_max: usize,
            word_max_size: usize, // in bytes, not len
            deduper: *Deduper(T),
        ) !Self {
            var arena_allocator = std.heap.ArenaAllocator.init(main_allocator);
            defer arena_allocator.deinit();
            const arena = arena_allocator.allocator();

            var word_edits = try std.ArrayListUnmanaged(WordEdit).initCapacity(arena, dict.len);
            var generator = try EditsGenerator(Word).init(arena, word_max_size);
            for (dict, 0..) |it, i| {
                try generator.load(it.word, wordMaxDistance(it.word), deduper);
                while (generator.next()) {
                    const str = try arena.dupe(T, generator.getValue());
                    try word_edits.append(arena, WordEdit{
                        .i_word = @intCast(i),
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
            // errdefer main_allocator.free(edits_index);

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
            const ef_edits_index = try EliasFano.init(main_allocator, edits_index[edits_index.len - 1], @TypeOf(&ef_iter), &ef_iter);

            return .{
                .pthash = ph,
                .edits_values = edits_values,
                .edits_index = ef_edits_index,
                .dict = dict,
            };
        }

        pub fn search(self: *const Self, allocator: std.mem.Allocator, word: []const T) ![]Hit {
            const len = wordLen(word);
            const max_distance = wordMaxDistance(word);

            const n = maxDeletes(len, max_distance);
            const bit_count = filter.bloomBitSize(n, bloom_fp_rate);
            const buffer_size = std.math.divCeil(usize, bit_count, 8) catch unreachable;
            var deduper = try BloomDeduper(u8).init(allocator, buffer_size, bloom_fp_rate);
            defer deduper.deinit();
            var generator = try EditsGenerator(Word).init(allocator, word.len);
            defer generator.deinit(allocator);
            try generator.load(word, max_distance, &deduper);

            var result = std.ArrayListUnmanaged(Hit){};
            errdefer result.deinit(allocator);

            var seen = std.AutoHashMapUnmanaged(u32, void){};
            defer seen.deinit(allocator);

            while (generator.next()) {
                const gen_token = generator.getValue();
                const hash = try self.pthash.get(gen_token);
                const d_ref = try self.edits_index.get(hash);
                {
                    const mm_hash = std.hash.Murmur3_32.hash(generator.getValue()) & 0x0000FFFF;
                    const mm_hash_chk = self.edits_values[d_ref] & 0x0000FFFF;
                    if (mm_hash != mm_hash_chk) continue;
                }
                const size = self.edits_values[d_ref] >> 16;

                for ((d_ref + 1)..(d_ref + 1 + size)) |i| {
                    const word_i = self.edits_values[i];
                    const it = self.dict[self.edits_values[i]];
                    if (@abs(@as(i64, @intCast(wordLen(it.word))) - @as(i64, @intCast(len))) > max_distance) {
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
                        });
                    }
                }
            }

            return result.toOwnedSlice(allocator);
        }

        // pub fn wordSegmentation(self: *const Self, allocator: std.mem.Allocator, word: []T) {
        //
        // }
    };
}

fn Deduper(comptime T: type) type {
    stringBackingTypeGuard(T);

    return struct {
        const Self = @This();
        pub const MAX_SIZE = 128;
        pub const VTable = struct {
            configure: *const fn (*anyopaque, len: usize, distance: usize) void,
            check: *const fn (*anyopaque, str: []const T) bool,
            put: *const fn (*anyopaque, str: []const T) void,
            deinit: *const fn (*anyopaque) void,
        };

        data: [MAX_SIZE]u8 align(@alignOf(u64)),
        vtable: VTable,

        pub fn init(comptime Container: type, value: Container) Self {
            comptime if (@sizeOf(Container) > MAX_SIZE) {
                @compileError("Type '" ++ @typeName(Container) ++ "' is too large, " ++ std.fmt.comptimePrint("{d} > {d}", .{ @sizeOf(Container), MAX_SIZE }));
            };

            var container = Self{
                .vtable = .{
                    .deinit = Container.deinit,
                    .configure = Container.configure,
                    .check = Container.check,
                    .put = Container.put,
                },
                .data = undefined,
            };

            const ptr: *Container = @ptrCast(@alignCast(&container.data));
            ptr.* = value;

            return container;
        }

        pub fn deinit(self: *Self) void {
            self.vtable.deinit(@ptrCast(&self.data));
        }

        pub fn configure(self: *Self, len: usize, distance: usize) void {
            self.vtable.configure(@ptrCast(&self.data), len, distance);
        }

        pub fn check(self: *Self, str: []const T) bool {
            return self.vtable.check(@ptrCast(&self.data), str);
        }

        pub fn put(self: *Self, str: []const T) void {
            self.vtable.put(@ptrCast(&self.data), str);
        }
    };
}

fn NoopDeduper(comptime T: type) type {
    stringBackingTypeGuard(T);

    return struct {
        const Self = @This();

        fn configure(_: *anyopaque, _: usize, _: usize) void {}
        fn put(_: *anyopaque, _: []const T) void {}
        fn deinit(_: *anyopaque) void {}
        fn check(_: *anyopaque, _: []const T) bool {
            return true;
        }

        pub fn init() Deduper(T) {
            return Deduper(T).init(Self, Self{});
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

        pub fn init(allocator: std.mem.Allocator) !Deduper(T) {
            const arena = try allocator.create(std.heap.ArenaAllocator);
            arena.child_allocator = allocator;
            arena.state = .{};

            const hash_map = try allocator.create(std.StringHashMapUnmanaged(void));
            hash_map.* = .{};
            return Deduper(T).init(Self, Self{
                .hash_map = hash_map,
                .allocator = allocator,
                .arena = arena,
            });
        }

        fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.arena.deinit();
            self.hash_map.deinit(self.allocator);
            self.allocator.destroy(self.arena);
            self.allocator.destroy(self.hash_map);
            self.* = undefined;
        }

        fn configure(ctx: *anyopaque, _: usize, _: usize) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            _ = self.arena.reset(.retain_capacity);
            self.hash_map.clearRetainingCapacity();
        }

        fn check(ctx: *anyopaque, word: []const T) bool {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.hash_map.contains(std.mem.sliceAsBytes(word));
        }

        fn put(ctx: *anyopaque, word: []const T) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
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

        pub fn init(allocator: std.mem.Allocator, buffer_size: usize, fp_rate: f64) !Deduper(T) {
            return Deduper(T).init(Self, Self{
                .filter = undefined,
                .fp_rate = fp_rate,
                .buffer = try allocator.alloc(u8, buffer_size),
                .allocator = allocator,
            });
        }

        fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.allocator.free(self.buffer);
            self.* = undefined;
        }

        fn configure(ctx: *anyopaque, len: usize, distance: usize) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
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

        fn check(ctx: *anyopaque, word: []const T) bool {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.filter.contains(std.mem.sliceAsBytes(word));
        }

        fn put(ctx: *anyopaque, word: []const T) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
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

        pub fn init(allocator: std.mem.Allocator, capacity: u32) !Deduper(T) {
            return Deduper(T).init(Self, Self{
                .filter = try filter.LRUFilter.init(allocator, capacity),
                .allocator = allocator,
            });
        }

        fn deinit(ctx: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.filter.deinit(self.allocator);
            self.* = undefined;
        }

        fn configure(ctx: *anyopaque, _: usize, _: usize) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.filter.clear();
        }

        fn check(ctx: *anyopaque, word: []const T) bool {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.filter.contains(std.mem.sliceAsBytes(word));
        }

        fn put(ctx: *anyopaque, word: []const T) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.filter.put(std.mem.sliceAsBytes(word));
        }
    };
}

fn EditsGenerator(comptime TStr: type) type {
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
        deduper: *Deduper(T),
        word_len: switch (TStr) {
            UTF8Buffer => usize,
            else => void,
        },

        pub fn init(allocator: std.mem.Allocator, buffer_size: usize) !Self {
            const deletes = try allocator.alloc(u8, buffer_size);
            errdefer allocator.free(deletes);

            const buffer = try allocator.alloc(T, buffer_size);
            errdefer allocator.free(buffer);

            return .{
                .word = undefined,
                .deduper = undefined,
                .max_distance = undefined,
                .word_len = undefined,
                .deletes = deletes,
                .buffer = buffer,
                .current_size = 0,
                .current_distance = 0,
            };
        }

        pub fn load(self: *Self, word: []const T, max_distance: usize, deduper: *Deduper(T)) !void {
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
                if (self.current_distance < self.max_distance) {
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

pub fn idrDistance(TStr: type, TBuf: type, dp: []TBuf, a: []const TStr, b: []const TStr) TBuf {
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

            var generator = try EditsGenerator([]const u8).init(allocator, word.len);

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

            var generator = try EditsGenerator(UTF8Buffer)
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

test "SymSpell" {
    const allocator = std.testing.allocator;
    const n = 500;
    var r = std.Random.DefaultPrng.init(0x4C27A681B1);
    const random = r.random();

    const Type = SymSpell([]const u8);

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

    std.debug.print("\n---- SymSpell - Search, query: {s} ----\n\n", .{query});

    for (hits, 0..) |hit, i| {
        std.debug.print("Hit[{}]: {s}, {}\n", .{ i, hit.word, hit.edit_distance });
    }
    try std.testing.expect(hits.len > 0);
    try std.testing.expectEqualStrings(hits[0].word, entries[0].word);
}
