const std = @import("std");
const filter = @import("filter.zig");
const pthash = @import("pthash.zig");
const util = @import("util.zig");
const iterator = @import("iterator.zig");
const ef = @import("ef.zig");

pub fn SymSpell(
    T: type,
) type {
    const wordDistance = struct {
        inline fn func(_: []T) usize {
            return 2;
        }
    }.func;

    const WordEdit = struct { i_word: u32, edit: []T };

    const EditsIterator = struct {
        array: []WordEdit,
        len: usize,
        i: usize = 0,

        pub fn size(self: *const @This()) usize {
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

    const PTHash = pthash.PTHash([]const T, pthash.OptimalMapper(u64), EditsIterator);
    const EliasFano = ef.GenericEliasFano(usize);

    return struct {
        const Self = @This();
        pub const Entry = struct {
            key: []T,
            count: u32,
        };

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

        pub fn init(main_allocator: std.mem.Allocator, items: []Entry) !Self {
            var arena = std.heap.ArenaAllocator.init(main_allocator);
            defer arena.deinit();
            const allocator = arena.allocator();

            var word_edits = try std.ArrayListUnmanaged(WordEdit)
                .initCapacity(allocator, items.len);

            const n = maxDeletes(50, 2);
            const fp_rate = 0.01;
            const bit_count = filter.bloomBitSize(n, fp_rate);
            const bloom_buffer = try allocator.alloc(u8, std.math.divCeil(usize, bit_count, 8) catch unreachable);
            defer allocator.free(bloom_buffer);

            const deduper = DeduperBloom([]u8).init(bloom_buffer, fp_rate);

            var generator = try EditsGenerator(T, @TypeOf(deduper))
                .init(allocator, 50);

            for (items, 0..) |it, i| {
                try generator.load(it.key, wordDistance(it.key), deduper);
                while (generator.next()) {
                    const str = try allocator.dupe(T, generator.getValue());
                    try word_edits.append(allocator, WordEdit{
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

            var iter = EditsIterator{ .array = word_edits.items, .len = unique };
            var ph = try PTHash.build(main_allocator, &iter, PTHash.buildConfig(iter.size(), .{
                .alpha = 0.97,
                .minimal = true,
            }));
            errdefer ph.deinit(allocator);

            var word_edits_groups = try allocator.alloc([]WordEdit, ph.size());
            @memset(word_edits_groups, &.{});

            var start_i: usize = 0;
            var prev_idx = try ph.get(word_edits.items[0].edit);

            for (1..word_edits.items.len) |i| {
                const hash = try ph.get(word_edits.items[i].edit);
                if (hash == prev_idx) continue;
                word_edits_groups[prev_idx] = word_edits.items[start_i..i];
                start_i = i;
                prev_idx = hash;
            }
            word_edits_groups[prev_idx] = word_edits.items[start_i..];

            var edits_index = try allocator.alloc(usize, ph.size());
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
            const eff = try EliasFano.init(main_allocator, edits_index[edits_index.len - 1], @TypeOf(&ef_iter), &ef_iter);

            return .{
                .pthash = ph,
                .edits_values = edits_values,
                .edits_index = eff,
            };
        }
    };
}

fn DeduperNoop(T: type) type {
    return struct {
        pub fn configure(_: *@This(), _: usize, _: usize) void {}
        pub fn check(_: *@This(), _: T) bool {
            return false;
        }
        pub fn put(_: *@This(), _: T) void {}
    };
}

fn DeduperHashSet(T: type) type {
    return struct {
        const Self = @This();

        hash_map: std.StringHashMapUnmanaged(void),
        allocator: std.mem.Allocator,
        arena: std.heap.ArenaAllocator,

        pub fn init(allocator: std.mem.Allocator) !Self {
            return .{
                .hash_map = std.StringHashMapUnmanaged(void){},
                .allocator = allocator,
                .arena = std.heap.ArenaAllocator.init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.arena.deinit();
            self.hash_map.deinit(self.allocator);
            self.* = undefined;
        }

        pub fn configure(self: *Self, _: usize, _: usize) void {
            _ = self.arena.reset(.retain_capacity);
            self.hash_map.clearRetainingCapacity();
        }

        pub fn check(self: *Self, word: T) bool {
            return self.hash_map.contains(word);
        }

        pub fn put(self: *Self, word: T) void {
            const duped_word = self.arena.allocator().dupe(u8, word) catch @panic(@typeName(Self) ++ " out of memory error");
            self.hash_map.put(self.allocator, duped_word, {}) catch @panic(@typeName(Self) ++ " out of memory error");
        }
    };
}

fn DeduperBloom(T: type) type {
    return struct {
        const Self = @This();

        fp_rate: f64,
        buffer: []u8,
        filter: filter.BloomFilter6,

        pub fn init(buffer: []u8, fp_rate: f64) Self {
            return .{
                .buffer = buffer,
                .fp_rate = fp_rate,
                .filter = undefined,
            };
        }

        pub fn configure(self: *Self, len: usize, distance: usize) void {
            const n = maxDeletes(len, distance);
            const bit_count = filter.bloomBitSize(n, self.fp_rate);
            const n_hash_fn = filter.bloomNHashFn(n, self.fp_rate);
            const buffer_size = std.math.divCeil(
                usize,
                @min(bit_count, self.buffer.len * 8),
                8,
            ) catch unreachable;
            self.filter = filter.BloomFilter6.initSlice(
                self.buffer[0..buffer_size],
                buffer_size * 8,
                @min(n_hash_fn, 6),
            ) catch unreachable;

            @memset(self.buffer[0..buffer_size], 0);
        }

        pub fn check(self: *Self, word: T) bool {
            return self.filter.contains(word);
        }

        pub fn put(self: *Self, word: T) void {
            self.filter.put(word);
        }
    };
}

fn EditsGenerator(T: type, Deduper: type) type {
    return struct {
        const Self = @This();

        max_distance: usize,
        current_distance: usize,
        current_size: usize,
        buffer: []T,
        word: []const T,
        deletes: []u8,
        deduper: Deduper,

        pub fn init(allocator: std.mem.Allocator, buffer_size: usize) !Self {
            const deletes = try allocator.alloc(u8, buffer_size);
            errdefer allocator.free(deletes);

            const buffer = try allocator.alloc(T, buffer_size);
            errdefer allocator.free(buffer);

            return .{
                .word = undefined,
                .deduper = undefined,
                .max_distance = undefined,
                .deletes = deletes,
                .buffer = buffer,
                .current_size = 0,
                .current_distance = 0,
            };
        }

        pub fn load(self: *Self, word: []const T, max_distance: usize, deduper: Deduper) !void {
            if (word.len > self.deletes.len) return error.Overflow;

            self.word = word;
            self.max_distance = max_distance;
            self.current_distance = 0;
            self.deduper = deduper;
            self.deduper.configure(word.len, max_distance);
            @memset(self.deletes[0..word.len], '_'); // printable, debuggability++
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.deletes);
            allocator.free(self.buffer);
            self.* = undefined;
        }

        pub fn getValue(self: *const Self) []T {
            return self.buffer[0..self.current_size];
        }

        pub fn next(self: *Self) bool {
            const n = self.word.len;
            const deletes = self.deletes[0..n];
            self.current_size = n - self.current_distance;
            const buffer = self.buffer[0..self.current_size];
            var j: usize = 0;
            for (0..n) |i| {
                if (deletes[i] == 'x') continue;
                buffer[j] = self.word[i];
                j += 1;
            }
            //std.debug.print("{s} {s}\n", .{ buffer, deletes });

            if (self.current_distance == 0 or !nextPermutation(T, deletes)) {
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

pub fn deduplicateStrings(T: type, strings: [][]T) [][]T {
    if (strings.len <= 1) return strings;
    std.mem.sort([]T, strings, {}, struct {
        fn func(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.func);

    var j: usize = 1;
    for (strings[1..]) |it| {
        if (!std.mem.eql(T, it, strings[j - 1])) {
            strings[j] = it;
            j += 1;
        }
    }

    return strings[0..j];
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

            var generator = try EditsGenerator(u8, @TypeOf(deduper))
                .init(allocator, word.len);

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
    const bloom_buffer = try allocator.alloc(u8, std.math.divCeil(usize, bit_count, 8) catch unreachable);
    defer allocator.free(bloom_buffer);

    const bloom_dp = DeduperBloom([]u8).init(bloom_buffer, fp_rate);
    try testFn(bloom_dp, true);

    var hs_dp = try DeduperHashSet([]u8).init(allocator);
    defer hs_dp.deinit();
    try testFn(&hs_dp, true);

    try testFn(DeduperNoop([]u8){}, false);
}

test "SymSpell" {
    const allocator = std.testing.allocator;
    const n = 500;
    var r = std.Random.DefaultPrng.init(0x4C27A681B1);
    const random = r.random();

    const Type = SymSpell(u8);

    var entries = try allocator.alloc(Type.Entry, n);
    defer {
        for (entries) |s| allocator.free(s.key);
        allocator.free(entries);
    }

    for (0..n) |i| {
        entries[i] = .{
            .key = try util.randomString(allocator, 5 + random.uintLessThan(usize, 20), random),
            .count = 1,
        };
    }

    var algo = try SymSpell(u8).init(allocator, entries);
    defer algo.deinit(allocator);
}
