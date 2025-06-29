const std = @import("std");
const bit = @import("bit.zig");
const ef = @import("ef.zig");
const util = @import("util.zig");
const iface = @import("interface.zig");
const iterator = @import("iterator.zig");

const toF64 = util.toF64;

/// Interface definition for bucket mappers
/// Intended only as a comptime reference for `interface.zig` -> `checkImplements`
fn IMapper(Hash: type) type {
    return struct {
        pub fn numBuckets(_: @This()) usize {
            unreachable;
        }
        pub fn getBucket(_: @This(), _: Hash) usize {
            unreachable;
        }
    };
}

/// Interface definition for hashers used in PTHash
/// Intended only as a comptime reference for `interface.zig` -> `checkImplements`
fn IHasher(Seed: type, Key: type, Hash: type) type {
    return struct {
        pub fn hashKey(_: @This(), _: Seed, _: Key) Hash {
            unreachable;
        }
        pub fn hashPilot(_: @This(), _: Seed, _: usize) Hash {
            unreachable;
        }
    };
}

/// Skewed mapper for PTHash, from original paper
pub fn SkewedMapper(Hash: type) type {
    return struct {
        const Self = @This();
        const a: f64 = 0.3;
        const b: f64 = 0.6;

        num_buckets: usize,
        num_keys: usize,
        p1: u64,
        p2: u64,

        pub fn init(num_keys: usize, num_buckets: usize) Self {
            const p1: u64 = @intFromFloat(a * toF64(num_buckets));
            return .{
                .num_buckets = num_buckets,
                .num_keys = num_keys,
                .p1 = p1,
                .p2 = num_buckets - p1,
            };
        }

        pub fn numBuckets(self: Self) usize {
            return self.num_buckets;
        }

        pub fn getBucket(self: Self, hash: Hash) usize {
            // S1 = { x | (h(x, seed) mod n) < p1 }
            // res = if (h(x, seed) ∈ S1) h(x, seed) mod p2 else p2 + h(x, seed) mod (m - p2)
            const p1_top: Hash = comptime @intFromFloat(toF64(std.math.maxInt(Hash)) * b);
            return if (hash < p1_top) hash % self.p1 else self.p1 + (hash % self.p2);
        }
    };
}

/// Optimal mapper for PTHash, from PTHash PHOBIC paper
pub fn OptimalMapper(Hash: type) type {
    return struct {
        const Self = @This();
        num_buckets: usize,
        num_keys: usize,
        eps: f64,
        map_constant: f64, // from hash to (0, 1)

        pub fn init(num_keys: usize, num_buckets: usize, eps: f64) Self {
            return .{
                .eps = eps,
                .num_buckets = num_buckets,
                .num_keys = num_keys,
                .map_constant = 1.0 / @as(f64, @floatFromInt(std.math.maxInt(u64))),
            };
        }

        pub fn numBuckets(self: Self) usize {
            return self.num_buckets;
        }

        pub fn getBucket(self: Self, hash: Hash) usize {
            const x = @as(f64, @floatFromInt(hash)) * self.map_constant;
            // beta_star(x) = x + (1 - x) * ln(1 - x)
            // beta_esp(x) = eps * x + (1 - eps) * beta_star(x)
            const one_minus_x = 1.0 - x;
            const beta_star = x + one_minus_x * @log(one_minus_x);
            const beta_eps = self.eps * x + (1.0 - self.eps) * beta_star;
            const bucket_f = beta_eps * @as(f64, @floatFromInt(self.num_buckets));
            return @intFromFloat(bucket_f);
        }
    };
}

/// Default hasher for string-like keys using Wyhash and Murmur
pub fn DefaultStringHasher(comptime Key: type) type {
    if (!util.isStringSlice(Key)) @compileError("Unsupported string type " ++ @typeName(Key));

    return struct {
        pub fn hashKey(_: @This(), seed: u64, key: Key) u64 {
            return std.hash.Wyhash.hash(seed, std.mem.sliceAsBytes(key));
        }

        pub fn hashPilot(_: @This(), seed: u64, pilot: usize) u64 {
            return std.hash.murmur.Murmur2_64.hashWithSeed(std.mem.asBytes(&pilot), seed);
        }
    };
}

/// Default hasher for int keys using Murmur
pub fn DefaultNumberHasher(comptime Key: type) type {
    if (@typeInfo(Key) != .int) @compileError("Unsupported int type " ++ @typeName(Key));

    return struct {
        pub fn hashKey(_: @This(), seed: u64, key: Key) u64 {
            return std.hash.murmur.Murmur2_64.hashWithSeed(std.mem.asBytes(&key), seed);
        }

        pub fn hashPilot(_: @This(), seed: u64, pilot: usize) u64 {
            return std.hash.murmur.Murmur2_64.hashWithSeed(std.mem.asBytes(&pilot), seed);
        }
    };
}

/// Configuration for PTHash construction
pub fn PTHashConfig(Mapper: type, Hasher: type) type {
    return struct {
        alpha: f64,
        minimal: bool,
        hashed_pilot_cache_size: usize,
        max_bucket_size: usize,
        mapper: Mapper,
        hasher: Hasher,
    };
}

/// Generic PTHash implementation (minimal perfect hash or compressed)
pub fn GenericPTHash(
    Seed: type,
    Hash: type,
    Key: type,
    Mapper: type,
    Hasher: type,
) type {
    iface.checkImplementsGuard(IMapper(Hash), Mapper);
    iface.checkImplementsGuard(IHasher(Seed, Key, Hash), Hasher);

    const BucketID = usize;

    const BucketHash = struct {
        bucket: BucketID,
        hash: Hash,
    };

    const Bucket = struct {
        bucket: BucketID,
        i_start: usize,
        i_end: usize, // exclusive
    };

    return struct {
        const Self = @This();

        seed: Seed,
        num_keys: usize,
        table_size: usize,
        config: PTHashConfig(Mapper, Hasher),
        pilots: ef.EliasFanoPS,
        free_slots: ?ef.EliasFano,

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.pilots.deinit(allocator);
            if (self.free_slots) |*fs| fs.deinit(allocator);
            self.* = undefined;
        }

        /// Builds a PTHash using random seed. Calls `buildSeed` internally
        pub fn build(
            allocator: std.mem.Allocator,
            comptime KeyIterator: type,
            keys: KeyIterator,
            config: PTHashConfig(Mapper, Hasher),
        ) !Self {
            iface.checkImplementsGuard(iterator.Iterator(Key), iface.UnPtr(KeyIterator));

            var l_seed: Seed = undefined;
            try std.posix.getrandom(std.mem.asBytes(&l_seed));
            return buildSeed(allocator, KeyIterator, keys, config, l_seed);
        }

        /// Returns the total size of the final table
        /// If `minimal` is true, this is equal to the number of input keys
        pub fn size(self: *const Self) usize {
            if (self.config.minimal) return self.num_keys;
            return self.table_size;
        }

        /// Builds PFH or MPFH using a fixed seed
        pub fn buildSeed(
            allocator: std.mem.Allocator,
            comptime KeyIterator: type,
            keys: KeyIterator,
            config: PTHashConfig(Mapper, Hasher),
            seed: Seed,
        ) !Self {
            iface.checkImplementsGuard(iterator.Iterator(Key), iface.UnPtr(KeyIterator));

            const hashes = try allocator.alloc(BucketHash, keys.size());
            defer allocator.free(hashes);
            {
                var i: usize = 0;
                while (keys.next()) |key| : (i += 1) {
                    const hash = config.hasher.hashKey(seed, key);
                    const bucket = config.mapper.getBucket(hash);
                    hashes[i] = BucketHash{ .hash = hash, .bucket = bucket };
                }
            }

            std.mem.sort(BucketHash, hashes, {}, struct {
                fn func(_: void, a: BucketHash, b: BucketHash) bool {
                    return if (a.bucket == b.bucket) a.hash < b.hash else a.bucket < b.bucket;
                }
            }.func);

            const max_bucket_size = config.max_bucket_size;
            const num_buckets = config.mapper.numBuckets();
            var table_size: usize = @intFromFloat(@ceil(@as(f64, @floatFromInt(keys.size())) / config.alpha));
            if (table_size & (table_size - 1) == 0) table_size += 1;

            const BucketList = std.ArrayListUnmanaged(Bucket);
            var buckets_by_size = try allocator.alloc(BucketList, max_bucket_size);
            defer {
                for (buckets_by_size) |*it| it.deinit(allocator);
                allocator.free(buckets_by_size);
            }
            @memset(buckets_by_size, BucketList{});

            var bucket_start: usize = 0;
            for (1..hashes.len) |i| {
                if (hashes[i].bucket == hashes[i - 1].bucket) {
                    if (hashes[i].hash == hashes[i - 1].hash) {
                        @branchHint(.unlikely);
                        return error.Collision;
                    }
                } else {
                    const len = (i - bucket_start);
                    if (len >= buckets_by_size.len) {
                        @branchHint(.unlikely);
                        return error.BucketOverflow;
                    }
                    try buckets_by_size[max_bucket_size - len - 1].append(allocator, Bucket{
                        .bucket = hashes[i - 1].bucket,
                        .i_start = bucket_start,
                        .i_end = i,
                    });
                    bucket_start = i;
                }
            }

            {
                const len = hashes.len - bucket_start;
                if (len > buckets_by_size.len) {
                    @branchHint(.unlikely);
                    return error.BucketOverflow;
                }
                try buckets_by_size[max_bucket_size - len - 1].append(allocator, Bucket{
                    .bucket = hashes[hashes.len - 1].bucket,
                    .i_start = bucket_start,
                    .i_end = hashes.len,
                });
            }

            var positions = try std.ArrayListUnmanaged(usize).initCapacity(allocator, max_bucket_size);
            defer positions.deinit(allocator);

            var table = try bit.BitArray.initCapacity(allocator, table_size);
            defer table.deinit(allocator);
            table.expandToCapacity();
            table.clearAll();

            var pilots = try allocator.alloc(u64, num_buckets);
            defer allocator.free(pilots);
            @memset(pilots, 0);

            var hashedPilotsCache = try allocator.alloc(Hash, config.hashed_pilot_cache_size);
            defer allocator.free(hashedPilotsCache);
            for (0..config.hashed_pilot_cache_size) |i| hashedPilotsCache[i] = config.hasher.hashPilot(seed, i);

            for (buckets_by_size) |buckets| {
                if (buckets.items.len == 0) continue;

                for (buckets.items) |bucket| {
                    var pilot: u64 = 0;
                    pilot_search: while (true) : (pilot += 1) {
                        const hashed_pilot = if (pilot < config.hashed_pilot_cache_size) hp: {
                            @branchHint(.likely);
                            break :hp hashedPilotsCache[pilot];
                        } else config.hasher.hashPilot(seed, pilot);

                        positions.clearRetainingCapacity();

                        for (hashes[bucket.i_start..bucket.i_end]) |it| {
                            const p = (it.hash ^ hashed_pilot) % table_size;
                            if (try table.get(u1, p) != 0) continue :pilot_search;
                            positions.appendAssumeCapacity(p);
                        }

                        std.mem.sort(usize, positions.items, {}, std.sort.asc(usize));
                        for (1..positions.items.len) |j| {
                            if (positions.items[j - 1] == positions.items[j]) continue :pilot_search;
                        }

                        pilots[bucket.bucket] = pilot;
                        for (positions.items) |p| table.set(p);
                        break;
                    }
                }
            }

            var iter_pilots = iterator.SliceIterator(u64).init(pilots);
            var enc_pilots = try ef.EliasFanoPS.init(allocator, 0, @TypeOf(&iter_pilots), &iter_pilots);
            errdefer enc_pilots.deinit(allocator);

            var enc_free_slots: ?ef.EliasFano = null;
            if (config.minimal) {
                var free_slots = try allocator.alloc(u64, table_size - hashes.len);
                defer allocator.free(free_slots);
                @memset(free_slots, 0);

                var prev: usize = 0;
                var i_free: usize = 0;
                var i_table: usize = 0;
                const n = hashes.len;
                while (i_free < free_slots.len) : (i_free += 1) {
                    if (table.isSet(n + i_free)) {
                        const idx = table.idxNext(i_table, 0).?;
                        i_table = idx + 1;
                        free_slots[i_free] = idx;
                        prev = idx;
                    } else {
                        free_slots[i_free] = prev;
                    }
                }

                var iter_fs = iterator.SliceIterator(u64).init(free_slots);
                enc_free_slots = try ef.EliasFano.init(allocator, free_slots[free_slots.len - 1], @TypeOf(&iter_pilots), &iter_fs);
                errdefer enc_free_slots.deinit(allocator);
            }

            return .{
                .seed = seed,
                .num_keys = hashes.len,
                .table_size = table_size,
                .config = config,
                .free_slots = enc_free_slots,
                .pilots = enc_pilots,
            };
        }

        /// Get the index of a given key in the table
        pub fn get(self: *const Self, key: Key) !u64 {
            const hash = self.config.hasher.hashKey(self.seed, key);
            const bucket = self.config.mapper.getBucket(hash);
            const pilot: u64 = try self.pilots.get(bucket);
            const hashed_pilot = self.config.hasher.hashPilot(self.seed, pilot);
            const p = (hash ^ hashed_pilot) % self.table_size;
            if (self.config.minimal and p >= self.num_keys) {
                return self.free_slots.?.get(p - self.num_keys);
            }
            return p;
        }
    };
}

/// Parameter struct with defaults for PTHash wrapper
pub const PTHashParams = struct {
    /// Average bucket size
    lambda: f64 = 6,
    /// Load factor
    alpha: f64 = 0.97,
    /// Whether to build a minimal perfect hash
    minimal: bool = false,
    /// Fixed number of buckets to use, overrides automatic estimation if set
    num_buckets: ?usize = null,
    /// Cache size for hashed pilot values during construction
    hashed_pilot_cache_size: usize = 4096,
    /// Maximum allowed size for any individual bucket
    max_bucket_size: usize = 255,
};

/// Wrapper to simplify PTHash instantiation
pub fn PTHash(
    comptime Key: type,
    comptime Mapper: type,
) type {
    if (Mapper != OptimalMapper(u64) and Mapper != SkewedMapper(u64)) {
        @compileError("Unknown mapper " ++ @typeInfo(Mapper) ++ ", please use GenericPTHash");
    }
    const Seed = u64;
    const Hasher = if (@typeInfo(Key) == .int) DefaultNumberHasher(Key) else DefaultStringHasher(Key);
    const T = GenericPTHash(Seed, u64, Key, Mapper, Hasher);

    return struct {
        pub const Type = T;

        pub fn buildConfig(num_keys: usize, params: PTHashParams) PTHashConfig(Mapper, Hasher) {
            const max_bucket_size = params.max_bucket_size;

            const num_buckets: usize = params.num_buckets orelse nb: {
                break :nb @intFromFloat(toF64(num_keys) / params.lambda);
            };

            const mapper = mp: {
                if (Mapper == SkewedMapper(u64)) {
                    break :mp SkewedMapper(u64).init(num_keys, num_buckets);
                }
                const eps: f64 = @max(0.0, @min(1.0, toF64(max_bucket_size) / (5.0 * std.math.sqrt(toF64(num_keys)))));
                break :mp OptimalMapper(u64).init(num_keys, num_buckets, eps);
            };

            return .{
                .alpha = params.alpha,
                .minimal = params.minimal,
                .hashed_pilot_cache_size = params.hashed_pilot_cache_size,
                .max_bucket_size = max_bucket_size,
                .mapper = mapper,
                .hasher = Hasher{},
            };
        }

        pub fn build(
            allocator: std.mem.Allocator,
            comptime KeyIterator: type,
            keys: KeyIterator,
            config: PTHashConfig(Mapper, Hasher),
        ) !T {
            return T.build(allocator, KeyIterator, keys, config);
        }

        pub fn buildSeed(
            allocator: std.mem.Allocator,
            comptime KeyIterator: type,
            keys: KeyIterator,
            config: PTHashConfig(Mapper, Hasher),
            seed: Seed,
        ) !T {
            return T.buildSeed(allocator, KeyIterator, keys, config, seed);
        }
    };
}

const testing = std.testing;

test "PTHash" {
    std.debug.print("\n----PTHash----\n\n", .{});

    const allocator = std.testing.allocator;
    var r = std.Random.DefaultPrng.init(0x4C27A681B1);
    const random = r.random();

    const n = 50000;
    var rs = try util.randomStrings(allocator, n, 30, random);
    defer rs.deinit(allocator);
    const data = rs.strings;

    const start = std.time.nanoTimestamp();

    var iter = iterator.SliceIterator([]const u8).init(data);
    const PTHashT = PTHash([]const u8, OptimalMapper(u64));
    var res = try PTHashT.buildSeed(allocator, @TypeOf(&iter), &iter, PTHashT.buildConfig(iter.size(), .{
        .lambda = 3,
        .alpha = 0.97,
        .minimal = false,
    }), 42);

    defer res.deinit(allocator);
    const diff = std.time.nanoTimestamp() - start;

    std.debug.print("Build: ns per key: {d:.2}\n", .{toF64(diff) / toF64(data.len)});

    var seen = try testing.allocator.alloc(bool, res.table_size);
    defer testing.allocator.free(seen);
    @memset(seen, false);

    var dataSize: u64 = 0;
    for (data) |key| {
        dataSize += key.len;
        const index = try res.get(key);
        try testing.expect(index < seen.len);
        try testing.expect(!seen[index]);
        seen[index] = true;
    }

    var count: usize = 0;
    for (seen) |b| {
        if (b) count += 1;
    }
    try testing.expect(count == n);

    const size = try util.calculateRuntimeSize(allocator, @TypeOf(res), res);

    var hb = util.HumanBytes{};

    std.debug.print(
        "Bits per key: {d:.4}\nPTHash size:  {s}\nTable size:   {}\nNum keys:     {}\nNum buckets:  {}\n",
        .{
            (toF64(size) * 8.0) / toF64(n), hb.fmt(size), res.table_size, n, res.config.mapper.numBuckets(),
        },
    );
}
