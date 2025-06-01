const std = @import("std");
const util = @import("util.zig");

// Dodging zig's strange fn pointer types
pub fn hashFnArray(comptime hashes: anytype) []@TypeOf(hashes[0]) {
    return @constCast(&hashes);
}

pub fn BloomFilter(
    comptime HashFunctions: []*const fn (*anyopaque, []const u8) u64,
) type {
    return struct {
        const Self = @This();

        bit_array: []u8,
        bit_count: usize,
        hash_fn_cnt: usize,
        owns_bit_array: bool,

        pub fn init(allocator: std.mem.Allocator, bit_count: usize, hash_fn_cnt: usize) !Self {
            if (bit_count == 0) return error.InvlidBitCount;
            if (hash_fn_cnt > HashFunctions.len) return error.InvalidHashFnCount;

            const byte_count = (bit_count + 7) / 8;
            const bit_array = try allocator.alloc(u8, byte_count);
            @memset(bit_array, 0);
            return Self{
                .bit_array = bit_array,
                .bit_count = bit_count,
                .hash_fn_cnt = if (hash_fn_cnt == 0) HashFunctions.len else hash_fn_cnt,
                .owns_bit_array = true,
            };
        }

        pub fn initSlice(bit_array: []u8, bit_count: usize, hash_fn_cnt: usize) !Self {
            if (bit_count == 0) return error.InvlidBitCount;
            if (hash_fn_cnt > HashFunctions.len) return error.InvalidHashFnCount;
            if (bit_count > bit_array.len * 8) return error.InvlidBitCount;
            @memset(bit_array, 0);
            return Self{
                .bit_array = bit_array,
                .bit_count = bit_count,
                .hash_fn_cnt = if (hash_fn_cnt == 0) HashFunctions.len else hash_fn_cnt,
                .owns_bit_array = false,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            if (self.owns_bit_array)
                allocator.free(self.bit_array);
            self.* = undefined;
        }

        fn setBit(self: *Self, i: usize) void {
            self.bit_array[i / 8] |= @as(u8, 1) << @intCast(i % 8);
        }

        fn getBit(self: *Self, i: usize) bool {
            return (self.bit_array[i / 8] & (@as(u8, 1) << @intCast(i % 8))) != 0;
        }

        pub fn put(self: *Self, key: []const u8) void {
            for (HashFunctions[0..self.hash_fn_cnt]) |hashFn| {
                const h = hashFn(@ptrCast(self), key) % self.bit_count;
                self.setBit(h);
            }
        }

        pub fn contains(self: *Self, key: []const u8) bool {
            for (HashFunctions[0..self.hash_fn_cnt]) |hashFn| {
                const h = hashFn(@ptrCast(self), key) % self.bit_count;
                if (!self.getBit(h)) return false;
            }
            return true;
        }
    };
}

// https://stackoverflow.com/questions/658439/how-many-hash-functions-does-my-bloom-filter-need
// The number of bits needed for given amount of keys and false positive rate
pub fn bloomBitSize(n_keys: usize, fp_rate: f64) u64 {
    return @max(1, @as(u64, @intFromFloat(-toF64(n_keys) * @log(fp_rate) / (std.math.ln2 * std.math.ln2))));
}

pub fn bloomByteSize(n_keys: usize, fp_rate: f64) u64 {
    const bit_count = bloomBitSize(n_keys, fp_rate);
    return std.math.divCeil(u64, bit_count, 8) catch unreachable;
}

// The number of hash functions we should use
pub fn bloomNHashFn(n_keys: usize, fp_rate: f64) u64 {
    return @max(1, @as(u64, @intFromFloat(toF64(bloomBitSize(n_keys, fp_rate)) / toF64(n_keys) * std.math.ln2)));
}

pub const Hashes = struct {
    fn hash1(_: *anyopaque, key: []const u8) u64 {
        return std.hash.Wyhash.hash(0x1baf584fef93eecf, key);
    }
    fn hash2(_: *anyopaque, key: []const u8) u64 {
        return std.hash.RapidHash.hash(0x1baf584fef93eecf, key);
    }
    fn hash3(_: *anyopaque, key: []const u8) u64 {
        return std.hash.CityHash64.hashWithSeed(key, 0xe001ca06d3f24547);
    }
    fn hash4(_: *anyopaque, key: []const u8) u64 {
        return std.hash.XxHash64.hash(0x4f74774e3856cc95, key);
    }
    fn hash5(_: *anyopaque, key: []const u8) u64 {
        return std.hash.Adler32.hash(key);
    }
    fn hash6(_: *anyopaque, key: []const u8) u64 {
        return std.hash.Fnv1a_64.hash(key);
    }
};

const hashes6 = [_]*const fn (*anyopaque, []const u8) u64{
    Hashes.hash1, Hashes.hash2, Hashes.hash3, Hashes.hash4, Hashes.hash5, Hashes.hash6,
};

pub const BloomFilter6 = BloomFilter(hashFnArray(hashes6));

test "BloomFilter" {
    const allocator = std.testing.allocator;
    var r = std.Random.DefaultPrng.init(0x4C27A681B1);
    const random = r.random();

    const n = 50;
    var rs = try util.randomStrings(allocator, n, 30, random);
    defer rs.deinit(allocator);

    var bf = try BloomFilter6.init(std.testing.allocator, 1024, 0);
    defer bf.deinit(std.testing.allocator);

    for (rs.strings) |s| {
        bf.put(s);
    }
    for (rs.strings) |s| {
        try std.testing.expect(bf.contains(s));
    }

    try std.testing.expect(!bf.contains("water"));
}

const toF64 = util.toF64;

test "BloomFilterEstimation" {
    const allocator = std.testing.allocator;
    var r = std.Random.DefaultPrng.init(0x4C27A681B1);
    const random = r.random();

    const n = 1000;
    var rs = try util.randomStrings(allocator, n, 30, random);
    defer rs.deinit(allocator);
    var rs_n = try util.randomStrings(allocator, n, 30, random);
    defer rs_n.deinit(allocator);

    const expected_fp_rate = 0.01;
    const bit_count = comptime bloomBitSize(n, expected_fp_rate);
    const n_hash_fn = comptime bloomNHashFn(n, expected_fp_rate);

    std.debug.print("\n----BloomFilterEstimation----\n\n", .{});
    std.debug.print("Num keys: {}\n", .{n});
    std.debug.print("Bit size: {}\n", .{bit_count});
    std.debug.print("Hash functions count: {}\n", .{n_hash_fn});
    std.debug.print("Expected FP rate: {d:4}\n", .{expected_fp_rate});

    var bf = try BloomFilter6.init(std.testing.allocator, bit_count, @min(n_hash_fn, 6));
    defer bf.deinit(std.testing.allocator);

    for (rs.strings) |s| bf.put(s);

    var false_positives: usize = 0;
    for (rs_n.strings) |s| {
        if (bf.contains(s)) false_positives += 1;
    }

    const actual_fp_rate: f64 = toF64(false_positives) / toF64(rs_n.strings.len);

    std.debug.print("Actual FP rate: {d:4}\n", .{actual_fp_rate});

    try std.testing.expect(actual_fp_rate <= expected_fp_rate * 2.0);
}
