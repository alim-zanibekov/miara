const std = @import("std");
const bitops = @import("bitops.zig");

/// BitArray backed by usize slice
pub const BitArray = GenericBitArray(usize);

/// Returns a bit array type backed by the given unsigned integer type
pub fn GenericBitArray(comptime T: type) type {
    if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned)
        @compileError("BitArrayT requires an unsigned integer as backing type, found " ++ @typeName(T));

    return struct {
        const Self = @This();
        const MemError = std.mem.Allocator.Error;
        const Error = error{IndexOutOfBounds};

        const TSize = usize;

        data: std.ArrayListUnmanaged(T),
        /// Size in bits
        bit_len: TSize,
        /// Capacity in bits
        capacity: TSize,

        /// Initializes an empty bit array with no allocation
        pub fn init() Self {
            return Self{
                .data = std.ArrayListUnmanaged(T){},
                .bit_len = 0,
                .capacity = 0,
            };
        }

        /// Initializes a bit array with a given number of bits preallocated
        pub fn initCapacity(allocator: std.mem.Allocator, num: TSize) MemError!Self {
            var self = Self{
                .data = std.ArrayListUnmanaged(T){},
                .bit_len = 0,
                .capacity = num,
            };
            if (num == 0) return self;
            try self.data.ensureTotalCapacity(allocator, (num - 1) / @bitSizeOf(T) + 1);
            return self;
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.data.deinit(allocator);
            self.* = undefined;
        }

        inline fn appendHelper(
            self: *Self,
            data: anytype,
            n: std.math.Log2IntCeil(@TypeOf(data)),
            comptime Ctx: type,
            ctx: Ctx,
            comptime inc: fn (Ctx) MemError!void,
        ) MemError!void {
            var remaining = n;
            while (remaining > 0) {
                const buffer_pos = self.bit_len % @bitSizeOf(T);
                const to_write = @min(@bitSizeOf(T) - buffer_pos, remaining);
                const mask = ~@as(T, 0) >> @intCast(@as(u16, @bitSizeOf(T)) - to_write);
                if (self.bit_len >= self.data.items.len * @bitSizeOf(T)) {
                    try inc(ctx);
                }
                const value = @as(T, @intCast((data >> @intCast(remaining - to_write) & mask)));
                self.data.items[self.data.items.len - 1] |= @intCast(value << @intCast(@as(u16, @bitSizeOf(T)) -| to_write -| buffer_pos));

                self.bit_len += to_write;
                remaining -= to_write;
            }
        }

        /// Appends `n` lsb of `data`, growing memory as needed
        pub fn append(self: *Self, allocator: std.mem.Allocator, data: anytype, n: std.math.Log2IntCeil(@TypeOf(data))) MemError!void {
            if (@typeInfo(@TypeOf(data)) != .int or @typeInfo(@TypeOf(n)).int.signedness != .unsigned)
                @compileError("BitArrayT.append requires an unsigned integer, found " ++ @TypeOf(data));

            const Ctx = struct { s: *Self, a: std.mem.Allocator };
            var ctx = Ctx{ .a = allocator, .s = self };

            try self.appendHelper(data, n, *Ctx, &ctx, struct {
                pub fn func(c: *Ctx) MemError!void {
                    try c.s.data.append(c.a, 0);
                }
            }.func);

            self.capacity = self.data.items.len * 8;
        }

        /// Appends `n` lsb of `data`, assuming capacity is sufficient
        pub fn appendAssumeCapacity(self: *Self, data: anytype, n: std.math.Log2IntCeil(@TypeOf(data))) void {
            if (@typeInfo(@TypeOf(data)) != .int or @typeInfo(@TypeOf(n)).int.signedness != .unsigned)
                @compileError("BitArrayT.append requires an unsigned integer, found " ++ @TypeOf(data));

            self.appendHelper(data, n, *Self, self, struct {
                pub fn func(s: *Self) MemError!void {
                    s.data.items.len += 1;
                    s.data.items[s.data.items.len - 1] = 0;
                }
            }.func) catch unreachable;
        }

        /// Sets all bits to zero
        pub fn clearAll(self: *Self) void {
            self.data.items.len = self.data.capacity;
            self.bit_len = self.capacity;
            @memset(self.data.items, 0);
        }

        /// Sets all bits to one
        pub fn setAll(self: *Self) void {
            self.data.items.len = self.data.capacity;
            self.bit_len = self.capacity;
            @memset(self.data.items, std.math.maxInt(T));
        }

        /// Returns @bitSizeOf(U) bits starting at bit `index`
        pub fn get(self: *const Self, comptime U: type, index: TSize) Error!U {
            var i = index / @bitSizeOf(T);
            var k = index % @bitSizeOf(T);
            if (@bitSizeOf(U) + index > self.bit_len) {
                return Error.IndexOutOfBounds;
            }

            const Y = std.math.Log2IntCeil(U);
            const aligh_shift = comptime if (@bitSizeOf(U) > @bitSizeOf(T)) @bitSizeOf(U) - @bitSizeOf(T) else 0;
            const TMax = comptime if (@bitSizeOf(U) > @bitSizeOf(T)) U else T;

            var remaining: Y = @bitSizeOf(U);
            var result: U = 0;
            while (remaining > 0) {
                const to_write = @min(@bitSizeOf(T) - k, remaining);
                const mask = bitops.shlWithOverflow(~@as(T, 0), @as(u16, @bitSizeOf(T)) - to_write);
                const val: TMax = @intCast(bitops.shlWithOverflow(self.data.items[i], k) & mask);
                result |= @intCast((val << aligh_shift) >> @intCast(@as(u16, @bitSizeOf(TMax)) - remaining));
                remaining -= to_write;
                i += 1;
                k = 0;
            }

            return result;
        }

        /// Returns `n` bits starting at bit `index` as `Out`
        pub fn getN(self: *const Self, comptime Out: type, n: std.math.Log2IntCeil(Out), i: TSize) Error!Out {
            return switch (n) {
                inline 1...@bitSizeOf(Out) => |k| @intCast(try self.get(
                    std.meta.Int(.unsigned, k),
                    i,
                )),
                else => return Error.IndexOutOfBounds,
            };
        }

        /// True if bit at index `i` is set
        pub fn isSet(self: *const Self, i: TSize) bool {
            const item = self.data.items[i / @bitSizeOf(T)];
            return item & (@as(T, 1) << @intCast(@bitSizeOf(T) - 1 - i % @bitSizeOf(T))) != 0;
        }

        /// Finds the next bit equal to `query` (0 or 1) starting from bit `start`
        pub fn idxNext(self: *const BitArray, start: TSize, comptime query: u1) ?TSize {
            if (start >= self.bit_len)
                return null;

            var i = start / @bitSizeOf(T);
            var k = start % @bitSizeOf(T);
            while (i < self.data.items.len) {
                const word = if (query == 1) self.data.items[i] else ~self.data.items[i];

                const masked_word = bitops.shlWithOverflow(word, k) >> @intCast(k);
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
        pub fn set(self: *Self, i: TSize) void {
            self.data.items[i / @bitSizeOf(T)] =
                self.data.items[i / @bitSizeOf(T)] | (@as(T, 1) << @intCast(@bitSizeOf(T) - 1 - i % @bitSizeOf(T)));
        }

        /// Sets bit at index `i` to zero
        pub fn clear(self: *Self, i: TSize) void {
            self.data.items[i / @bitSizeOf(T)] =
                self.data.items[i / @bitSizeOf(T)] & ~(@as(T, 1) << @intCast(@bitSizeOf(T) - 1 - i % @bitSizeOf(T)));
        }
    };
}

test "bitArrayAppendAlloc" {
    const seed = 0x224740f963;
    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var c = BitArray.init();
    defer c.deinit(std.testing.allocator);
    const Y = usize;

    for (0..1000) |_| {
        try c.append(std.testing.allocator, random.int(Y), @bitSizeOf(Y));
    }

    r = std.Random.DefaultPrng.init(seed);
    random = r.random();

    var number: Y = 0;
    for (0..(1000 * @bitSizeOf(Y))) |i| {
        if (i % @bitSizeOf(Y) == 0) number = random.int(Y);
        const v = number & (@as(Y, 1) << @intCast(@bitSizeOf(Y) - 1 - (i % @bitSizeOf(Y)))) != 0;
        try std.testing.expectEqual(v, c.isSet(i));
    }
}

test "bitArrayInitAlloc" {
    const seed = 0x224740f963;
    const n = 1000;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    const Y = usize;
    var c = try BitArray.initCapacity(std.testing.allocator, n * @bitSizeOf(Y));
    defer c.deinit(std.testing.allocator);

    var numbers = try std.ArrayList(Y).initCapacity(std.testing.allocator, n);
    defer numbers.deinit();

    for (0..n) |_| numbers.appendAssumeCapacity(random.int(Y));

    for (numbers.items) |x| {
        const to_add = random.uintAtMost(u16, @bitSizeOf(Y) - 1) + 1;
        const remaining = @bitSizeOf(Y) - to_add;
        c.appendAssumeCapacity(x >> @intCast(remaining), @intCast(to_add));
        if (remaining != 0) {
            c.appendAssumeCapacity(x, @intCast(remaining));
        }
    }

    for (0..(numbers.items.len * @bitSizeOf(Y))) |i| {
        const number = numbers.items[i / @bitSizeOf(Y)];
        const v = number & (@as(Y, 1) << @intCast(@bitSizeOf(Y) - 1 - (i % @bitSizeOf(Y)))) != 0;
        try std.testing.expectEqual(v, c.isSet(i));
    }
}

test "bitArrayCustomBufferType" {
    const testFn = struct {
        pub fn func(comptime U: type, comptime seed: comptime_int) !void {
            const n = 1000;
            var r = std.Random.DefaultPrng.init(seed);
            var random = r.random();

            const Y = usize;
            var c = try GenericBitArray(U).initCapacity(std.testing.allocator, n * @bitSizeOf(Y));
            defer c.deinit(std.testing.allocator);

            var numbers = try std.ArrayList(Y).initCapacity(std.testing.allocator, n);
            defer numbers.deinit();

            for (0..n) |_| numbers.appendAssumeCapacity(random.int(Y));

            for (numbers.items) |x| {
                const toAdd = random.uintAtMost(u16, @bitSizeOf(Y) - 1) + 1;
                const remaining = @bitSizeOf(Y) - toAdd;
                c.appendAssumeCapacity(x >> @intCast(remaining), @intCast(toAdd));
                if (remaining != 0) {
                    c.appendAssumeCapacity(x, @intCast(remaining));
                }
            }

            for (0..(n * @bitSizeOf(Y))) |i| {
                const number = numbers.items[i / @bitSizeOf(Y)];
                const v = number & (@as(Y, 1) << @intCast(@bitSizeOf(Y) - 1 - (i % @bitSizeOf(Y)))) != 0;
                try std.testing.expectEqual(v, c.isSet(i));
            }

            for (numbers.items, 0..) |num, i| {
                try std.testing.expectEqual(num, c.get(Y, i * @bitSizeOf(Y)));
            }

            try std.testing.expectError(error.IndexOutOfBounds, c.get(Y, 1000 * @bitSizeOf(Y)));
        }
    }.func;

    try testFn(u69, 0x3C0633A9CA);
    try testFn(u3, 0x3C0633A9BA);
    try testFn(u1, 0x3C0633A9C1);
    try testFn(u64, 0x3C0633A9C2);
    try testFn(u181, 0x3C0633A9C9);
}
