const std = @import("std");

/// BitArray backed by usize slice
pub const BitArray = GenericBitArray(usize);

/// Returns a bit array type backed by the given unsigned integer type
pub fn GenericBitArray(comptime T: type) type {
    if (!std.meta.hasUniqueRepresentation(T)) {
        @compileError("GenericBitArray backing type must have no unused bits and padding, found " ++ @typeName(T));
    }

    return struct {
        pub const Type = T;

        const Self = @This();
        const BitArrayError = error{IndexOutOfBounds};
        const MemError = std.mem.Allocator.Error;

        const Size = usize;

        data: []T,
        /// Size in bits
        len: Size,

        /// Initializes an empty bit array with no allocation
        pub fn init() Self {
            return Self{
                .data = &[_]T{},
                .len = 0,
            };
        }

        /// Initializes a bit array with a given number of bits preallocated
        pub fn initCapacity(allocator: std.mem.Allocator, capacity: Size) MemError!Self {
            if (capacity == 0) return init();
            const capacity_container = @divFloor(capacity - 1, @bitSizeOf(T)) + 1;
            return Self{
                .data = try allocator.alloc(T, capacity_container),
                .len = 0,
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.data);
            self.* = undefined;
        }

        pub fn expandToCapacity(self: *Self) void {
            self.len = self.data.len * @bitSizeOf(T);
        }

        pub fn ensureCapacity(self: *Self, allocator: std.mem.Allocator, capacity: usize) !void {
            const growth_factor = 2;
            if (capacity > self.data.len * @bitSizeOf(T)) {
                const capacity_container = @divFloor(capacity - 1, @bitSizeOf(T)) + 1;

                const new_capacity = @max(capacity_container, self.data.len * growth_factor);
                self.data = try allocator.realloc(self.data, new_capacity);
            }
        }

        pub fn setRange(
            self: *Self,
            index: Size,
            data: anytype,
            from: std.math.IntFittingRange(0, @bitSizeOf(@TypeOf(data))),
            to: std.math.IntFittingRange(0, @bitSizeOf(@TypeOf(data))),
            endian: std.builtin.Endian,
        ) void {
            std.debug.assert(to >= from);
            std.debug.assert((index + (to - from) - 1) / @bitSizeOf(T) + 1 <= self.data.len);
            if (to == from) return;

            const bytes = std.mem.asBytes(&data);
            var i = from;
            var pos = index;
            while (i < to) {
                const dst_shift: std.math.Log2Int(T) = @intCast(pos % @bitSizeOf(T));
                const src_shift = i % 8;
                const drop_last: u3 = @intCast((i + 8) -| to);
                // @bitSizeOf(T) >= 8
                const byte = switch (endian) {
                    .big => bytes[i / 8],
                    .little => bytes[bytes.len - 1 - (i / 8)],
                };

                const value: T = (@as(T, @intCast((shl(byte, src_shift) >> drop_last) << drop_last)) << (@bitSizeOf(T) - 8)) >> dst_shift;
                const mask: T = (@as(T, @intCast((shl(@as(u8, 0xFF), src_shift) >> drop_last) << drop_last)) << (@bitSizeOf(T) - 8)) >> dst_shift;

                std.debug.assert(mask != 0);

                const j = pos / @bitSizeOf(T);
                self.data[j] = (self.data[j] & ~mask) | value;

                pos += @popCount(mask);
                i += @intCast(@popCount(mask)); // max 8
            }
        }

        /// Appends `n` lsb of `data`, growing memory as needed
        pub fn appendUInt(self: *Self, allocator: std.mem.Allocator, data: anytype, n: std.math.Log2IntCeil(@TypeOf(data))) MemError!void {
            if (@typeInfo(@TypeOf(data)) != .int or @typeInfo(@TypeOf(n)).int.signedness != .unsigned)
                @compileError("BitArrayT.append requires an unsigned integer, found " ++ @TypeOf(data));

            const bitSize: Size = @bitSizeOf(@TypeOf(data));
            try self.ensureCapacity(allocator, self.len + n);
            self.setRange(self.len, data, @intCast(bitSize - n), bitSize, .little);
            self.len += n;
        }

        /// Appends `n` lsb of `data`, assuming capacity is sufficient
        pub fn appendUIntAssumeCapacity(self: *Self, data: anytype, n: std.math.Log2IntCeil(@TypeOf(data))) void {
            if (@typeInfo(@TypeOf(data)) != .int or @typeInfo(@TypeOf(n)).int.signedness != .unsigned)
                @compileError("BitArrayT.append requires an unsigned integer, found " ++ @TypeOf(data));

            const bitSize: Size = @bitSizeOf(@TypeOf(data));
            self.setRange(self.len, data, @intCast(bitSize - n), bitSize, .little);
            self.len += n;
        }

        /// Sets all bits to zero
        pub fn clearAll(self: *Self) void {
            if (self.len == 0) return;
            @memset(self.data[0..(@divFloor(self.len - 1, @bitSizeOf(T)) + 1)], 0);
        }

        /// Sets all bits to one
        pub fn setAll(self: *Self) void {
            if (self.len == 0) return;
            @memset(self.data[0..(@divFloor(self.len - 1, @bitSizeOf(T)) + 1)], std.math.maxInt(T));
        }

        /// Returns @bitSizeOf(Out) bits starting at bit `index`
        pub fn get(self: *const Self, comptime Out: type, index: Size) BitArrayError!Out {
            if (@typeInfo(Out) != .int or @typeInfo(Out).int.signedness != .unsigned)
                @compileError("BitArray.get requires an unsigned integer as a result type, found " ++ @typeName(Out));

            const High = comptime if (@bitSizeOf(Out) > @bitSizeOf(T)) Out else T;
            const align_shift = comptime if (@bitSizeOf(Out) > @bitSizeOf(T)) @bitSizeOf(Out) - @bitSizeOf(T) else 0;

            var i = index / @bitSizeOf(T);
            var k = index % @bitSizeOf(T);

            if (@bitSizeOf(Out) + index > self.len) {
                return BitArrayError.IndexOutOfBounds;
            }

            const Log2T = std.math.Log2IntCeil(T);
            const Log2High = std.math.Log2IntCeil(High);
            var remaining: std.math.Log2IntCeil(Out) = @bitSizeOf(Out);
            var result: Out = 0;
            while (remaining > 0) {
                const to_write = @min(@bitSizeOf(T) - k, remaining);
                const mask = shl(~@as(T, 0), @as(Log2T, @bitSizeOf(T)) - to_write);
                const val: High = @intCast(shl(self.data[i], k) & mask);
                result |= @intCast((val << align_shift) >> @intCast(@as(Log2High, @bitSizeOf(High)) - remaining));
                remaining -= to_write;
                i += 1;
                k = 0;
            }
            return result;
        }

        /// Returns `n` bits starting at bit `index` as `T`
        pub fn getVarFast(self: *const Self, index: Size, n: std.math.Log2IntCeil(T)) T {
            std.debug.assert(n <= @bitSizeOf(T));

            const i = index / @bitSizeOf(T);
            const k = index % @bitSizeOf(T);
            const mask: T = ~@as(T, 0) >> @intCast(k);
            if (k + n <= @bitSizeOf(T)) {
                return (self.data[i] & mask) >> @intCast(@bitSizeOf(T) - (k + n));
            } else {
                return ((self.data[i] & mask) << @intCast((k + n) - @bitSizeOf(T))) | (self.data[i + 1] >> @intCast(@bitSizeOf(T) * 2 - (k + n)));
            }
        }

        /// Returns `n` bits starting at bit `index` as `Out`
        pub fn getVar(self: *const Self, comptime Out: type, n: std.math.Log2IntCeil(Out), i: Size) BitArrayError!Out {
            return switch (n) {
                inline 1...@bitSizeOf(Out) => |k| @intCast(try self.get(
                    std.meta.Int(.unsigned, k),
                    i,
                )),
                else => return BitArrayError.IndexOutOfBounds,
            };
        }

        /// True if bit at index `i` is set
        pub fn isSet(self: *const Self, i: Size) bool {
            const item = self.data[i / @bitSizeOf(T)];
            return item & (@as(T, 1) << @intCast(@bitSizeOf(T) - 1 - i % @bitSizeOf(T))) != 0;
        }

        /// Finds the next bit equal to `query` (0 or 1) starting from bit `start`
        pub fn idxNext(self: *const Self, start: Size, comptime query: u1) ?Size {
            if (start >= self.len)
                return null;

            var i = start / @bitSizeOf(T);
            var k = start % @bitSizeOf(T);

            while (i < self.data.len) {
                const word = if (query == 1) self.data[i] else ~self.data[i];

                const masked_word = std.math.shl(T, word, k) >> @intCast(k);
                if (masked_word != 0) {
                    const tz = @clz(masked_word);
                    const bit_index = i * @bitSizeOf(T) + tz;
                    return bit_index;
                }
                i += 1;
                k = 0;
            }

            return null;
        }

        /// Sets bit at index `i` to one
        pub fn set(self: *Self, i: Size) void {
            self.data[i / @bitSizeOf(T)] =
                self.data[i / @bitSizeOf(T)] | (@as(T, 1) << @intCast(@bitSizeOf(T) - 1 - i % @bitSizeOf(T)));
        }

        /// Sets bit at index `i` to zero
        pub fn clear(self: *Self, i: Size) void {
            self.data[i / @bitSizeOf(T)] =
                self.data[i / @bitSizeOf(T)] & ~(@as(T, 1) << @intCast(@bitSizeOf(T) - 1 - i % @bitSizeOf(T)));
        }
    };
}

pub inline fn shl(it: anytype, n: usize) @TypeOf(it) {
    if (n >= @bitSizeOf(@TypeOf(it))) return 0;
    return @shlWithOverflow(it, @as(std.math.Log2Int(@TypeOf(it)), @intCast(n)))[0];
}

pub const NthSetBitError = error{ InvalidN, NotFound };

const testing = std.testing;

/// Returns the index (0-based, from msb) of the n-th set bit in `src`
/// Requires `src` to be an unsigned integer
pub fn nthSetBitPos(src: anytype, n: std.math.Log2IntCeil(@TypeOf(src))) NthSetBitError!std.math.Log2Int(@TypeOf(src)) {
    const T = @TypeOf(src);
    const info = @typeInfo(T);
    if (info != .int or info.int.signedness != .unsigned)
        @compileError("nthSetBitPos requires an unsigned integer, found " ++ @typeName(T));

    if (n == 0 or n > info.int.bits) return NthSetBitError.InvalidN;

    if (comptime info.int.bits > 8 and std.math.isPowerOfTwo(info.int.bits)) {
        var bits = @bitReverse(src);
        var count: usize = n;
        for (0..(info.int.bits / 8)) |i| {
            const b = @popCount(bits & 0xff);
            if (count <= b) {
                const byte: u8 = @intCast(bits & 0xff);
                for (0..8) |j| {
                    if (byte >> @intCast(j) & 1 == 1) {
                        count -= 1;
                        if (count == 0) return @intCast((i << 3) + j);
                    }
                    bits >>= 1;
                }
            } else {
                count -= b;
                bits >>= 8;
            }
        }

        return error.NotFound;
    }

    var bits = @bitReverse(src);
    var count: usize = n;
    inline for (0..info.int.bits) |i| {
        if (bits & 1 == 1) {
            count -= 1;
            if (count == 0) return @intCast(i);
        }
        bits >>= 1;
    }

    return error.NotFound;
}

test nthSetBitPos {
    const expectEqual = std.testing.expectEqual;
    const expectError = std.testing.expectError;

    try expectEqual(3, try nthSetBitPos(@as(u8, 0b00010110), 1));
    try expectEqual(5, try nthSetBitPos(@as(u8, 0b00010110), 2));
    try expectEqual(6, try nthSetBitPos(@as(u8, 0b00010110), 3));

    try expectEqual(0, try nthSetBitPos(@as(u64, 1) << 63, 1));

    try expectEqual(63, try nthSetBitPos(@as(u64, 1), 1));

    for (1..65) |n| {
        try expectEqual(n - 1, try nthSetBitPos(~@as(u64, 0), @intCast(n)));
    }

    try expectError(NthSetBitError.InvalidN, nthSetBitPos(@as(u8, 0b10110), 0));
    try expectError(NthSetBitError.NotFound, nthSetBitPos(@as(u8, 0b10110), 4));
    try expectError(NthSetBitError.NotFound, nthSetBitPos(@as(u64, 0), 1));
}

test "bitArrayAppendAlloc" {
    const seed = 0x224740f963;
    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var c = BitArray.init();
    defer c.deinit(testing.allocator);
    const Y = usize;

    for (0..1000) |_| {
        try c.appendUInt(testing.allocator, random.int(Y), @bitSizeOf(Y));
    }

    r = std.Random.DefaultPrng.init(seed);
    random = r.random();

    var number: Y = 0;
    for (0..(1000 * @bitSizeOf(Y))) |i| {
        if (i % @bitSizeOf(Y) == 0) number = random.int(Y);
        const v = number & (@as(Y, 1) << @intCast(@bitSizeOf(Y) - 1 - (i % @bitSizeOf(Y)))) != 0;
        try testing.expectEqual(v, c.isSet(i));
    }
}

test "bitArrayInitAlloc" {
    const seed = 0x224740f963;
    const n = 1000;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    const T = usize;
    var array = try BitArray.initCapacity(testing.allocator, n * @bitSizeOf(T));
    defer array.deinit(testing.allocator);

    var numbers = try std.ArrayList(T).initCapacity(testing.allocator, n);
    defer numbers.deinit();

    for (0..n) |_| numbers.appendAssumeCapacity(random.int(T));

    for (numbers.items) |x| {
        const to_add = random.uintAtMost(u16, @bitSizeOf(T) - 1) + 1;
        const remaining = @bitSizeOf(T) - to_add;
        array.appendUIntAssumeCapacity(x >> @intCast(remaining), @intCast(to_add));
        if (remaining != 0) {
            array.appendUIntAssumeCapacity(x, @intCast(remaining));
        }
    }

    for (numbers.items, 0..) |number, pos| {
        const i = pos * @bitSizeOf(T);
        const v = number & (@as(T, 1) << @intCast(@bitSizeOf(T) - 1 - (i % @bitSizeOf(T)))) != 0;
        try testing.expectEqual(v, array.isSet(i));
    }
}

test "bitArrayCustomBufferType" {
    const testFn = struct {
        pub fn func(comptime U: type, comptime seed: comptime_int) !void {
            const n = 1000;
            var r = std.Random.DefaultPrng.init(seed);
            var random = r.random();

            const Y = usize;
            var c = try GenericBitArray(U).initCapacity(testing.allocator, n * @bitSizeOf(Y));
            defer c.deinit(testing.allocator);

            var numbers = try std.ArrayList(Y).initCapacity(testing.allocator, n);
            defer numbers.deinit();

            for (0..n) |_| numbers.appendAssumeCapacity(random.int(Y));

            for (numbers.items) |x| {
                const toAdd = random.uintAtMost(u16, @bitSizeOf(Y) - 1) + 1;
                const remaining = @bitSizeOf(Y) - toAdd;
                c.appendUIntAssumeCapacity(x >> @intCast(remaining), @intCast(toAdd));
                if (remaining != 0) {
                    c.appendUIntAssumeCapacity(x, @intCast(remaining));
                }
            }

            for (0..(n * @bitSizeOf(Y))) |i| {
                const number = numbers.items[i / @bitSizeOf(Y)];
                const v = number & (@as(Y, 1) << @intCast(@bitSizeOf(Y) - 1 - (i % @bitSizeOf(Y)))) != 0;
                try testing.expectEqual(v, c.isSet(i));
            }

            for (numbers.items, 0..) |num, i| {
                try testing.expectEqual(num, c.get(Y, i * @bitSizeOf(Y)));
            }

            try testing.expectError(error.IndexOutOfBounds, c.get(Y, 1000 * @bitSizeOf(Y)));
        }
    }.func;

    try testFn(u8, 0x3C0633A9CA);
    try testFn(u16, 0x3C0633A9BA);
    try testFn(u32, 0x3C0633A9C1);
    try testFn(u64, 0x3C0633A9C2);
}
