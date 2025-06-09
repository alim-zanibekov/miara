const std = @import("std");
const s = @import("spider.zig");
const bit = @import("bit.zig");
const iterator = @import("iterator.zig");
const iface = @import("interface.zig");

pub const EliasFano = GenericEliasFano(u64);

pub fn GenericEliasFano(T: type) type {
    if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned)
        @compileError("BitArrayT requires an unsigned integer as backing type, found " ++ @typeName(T));

    return struct {
        const Self = @This();
        pub const Error = error{ OutOfBounds, Undefined };

        higher_bits: s.Spider,
        lower_bits: bit.BitArray,
        low_n: u16,
        len: usize,

        pub fn init(allocator: std.mem.Allocator, universe: T, TIter: type, iter: TIter) !Self {
            iface.checkImplementsGuard(iterator.Iterator(T), iface.UnPtr(TIter));

            const m = universe;
            const n = iter.size();

            const log_m = std.math.log2_int_ceil(T, m);
            const log_n = std.math.log2_int_ceil(usize, n);

            const low_n = if (log_m > log_n) log_m - log_n else 0;
            const lb_n = n * low_n;
            const hb_n = (m >> @intCast(low_n)) + n;

            var hb = try s.Spider.Builder.initWithTotalCapacity(allocator, hb_n);
            errdefer hb.deinit(allocator);
            var lb = try bit.BitArray.initCapacity(allocator, lb_n);
            errdefer lb.deinit(allocator);

            var bucket: T = 0;
            var max: T = 0;
            while (iter.next()) |it| {
                max = @max(max, it);
                if (max != it) {
                    return error.NonMonotomicallyIncreasing;
                }
                const higher_bits = it >> @intCast(low_n);
                while (higher_bits > bucket) {
                    try hb.append(@as(u1, 0), 1);
                    bucket += 1;
                }
                try hb.append(@as(u1, 1), 1);
                lb.appendAssumeCapacity(it, low_n);
            }

            const spider = try hb.build(allocator);

            return Self{
                .lower_bits = lb,
                .higher_bits = spider,
                .low_n = low_n,
                .len = iter.size(),
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            self.higher_bits.deinit(allocator);
            self.lower_bits.deinit(allocator);
        }

        // Get i-th element
        pub fn get(self: *const Self, i: usize) !T {
            const h_bits = try self.higher_bits.select1(i + 1) - i;
            const l_bits = if (self.low_n > 0) try self.lower_bits.getRt(T, @intCast(self.low_n), self.low_n * i) else 0;
            return (h_bits << @intCast(self.low_n)) | l_bits;
        }

        // Get next greater or equal than `num`
        pub fn getNextGEQ(self: *const Self, num: T) !T {
            const i = num >> @intCast(self.low_n);
            var pos = if (i > 0) try self.higher_bits.select0(i) - i else 0;
            var result = try self.get(pos);
            while (result < num and pos < self.len) : (pos += 1) {
                result = try self.get(pos);
            }
            return if (result < num) s.Spider.Error.NotFound else result;
        }
    };
}

const testing = std.testing;

test "eliasFanoCodec" {
    const n = 500;
    const seed = 0x4C27A681B1;
    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try testing.allocator.alloc(u64, n);
    defer testing.allocator.free(data);

    data[0] = random.uintLessThan(u64, ~@as(u32, 0) / n);
    for (1..n) |i| {
        data[i] = data[i - 1] + random.uintLessThan(u64, ~@as(u32, 0) / n);
    }

    var iter = iterator.SliceIterator(u64).init(data);
    var ef = try EliasFano.init(testing.allocator, data[data.len - 1], @TypeOf(&iter), &iter);
    defer ef.deinit(testing.allocator);

    for (data, 0..) |it, i| {
        try testing.expectEqual(it, ef.get(i));
    }

    for (0..n - 2) |i| {
        const goq = try ef.getNextGEQ(data[i]);
        try testing.expect(data[i + 1] == goq or data[i] == goq);
    }
}
