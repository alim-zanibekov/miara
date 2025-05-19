const std = @import("std");
const builtin = @import("builtin");
const bitops = @import("bitops.zig");

const Spider = struct {
    pub const Error = error{ OutOfBounds, Undefined };

    const sb_size = 63488;
    const Self = @This();

    bit_array: []align(64) const u8,

    hl_rank: []const u64,
    hl_select: []const u64,
    ll_select: []const u16,

    sigma_hl_pow: u6,
    sigma_ll_pow: u4,

    size: u64,

    pub fn init(allocator: std.mem.Allocator, bits: []const u8, num_bits: u64) !Self {
        if (num_bits == 0) return Error.Undefined;
        if ((num_bits - 1) >> 3 + 1 > bits.len) return Error.OutOfBounds;

        const rnk = try build_rank(allocator, bits, num_bits);
        const sel = try build_select(allocator, rnk.count, num_bits, rnk.bit_array);

        return Spider{
            .hl_rank = rnk.hl_rank,
            .bit_array = rnk.bit_array,
            .hl_select = sel.hl_select,
            .ll_select = sel.ll_select,
            .sigma_hl_pow = sel.sigma_hl_pow,
            .sigma_ll_pow = sel.sigma_ll_pow,

            .size = num_bits,
        };
    }

    fn alloc_rank_arrays(allocator: std.mem.Allocator, num_bits: u64) !struct {
        std.ArrayListUnmanaged(u64),
        std.ArrayListAlignedUnmanaged(u8, 64),
    } {
        const n_hl = (num_bits - 1) / sb_size + 2;
        const n_rank = ((num_bits - 1) / (sb_size >> 7) + 1) * 64 + 2;

        var hl_rank = std.ArrayListUnmanaged(u64){};
        errdefer hl_rank.deinit(allocator);

        var bit_array = std.ArrayListAlignedUnmanaged(u8, 64){};
        errdefer bit_array.deinit(allocator);

        try hl_rank.ensureTotalCapacityPrecise(allocator, n_hl);
        try bit_array.ensureTotalCapacityPrecise(allocator, n_rank);

        return .{ hl_rank, bit_array };
    }

    inline fn rank_byte(
        byte: u8,
        k: usize,
        cumulative_count: *u64,
        relative_count: *u16,
        hl_rank: *std.ArrayListUnmanaged(u64),
        bit_array: *std.ArrayListAlignedUnmanaged(u8, 64),
    ) void {
        if (k % (sb_size >> 3) == 0) { // k % 7936
            hl_rank.appendAssumeCapacity(cumulative_count.*);
            relative_count.* = 0;
        }

        if (k % (sb_size >> 10) == 0) { // k % 62
            bit_array.items.len += 2;
            std.mem.writeInt(u16, bit_array.items[(bit_array.items.len - 2)..][0..2], relative_count.*, builtin.cpu.arch.endian());
        }

        cumulative_count.* += @popCount(byte);
        relative_count.* += @popCount(byte);

        bit_array.items.len += 1;
        bit_array.items[bit_array.items.len - 1] = byte;
    }

    inline fn rank_last_byte(
        remaining: u4,
        cumulative_count: *u64,
        hl_rank: *std.ArrayListUnmanaged(u64),
        bit_array: *std.ArrayListAlignedUnmanaged(u8, 64),
    ) void {
        const i = bit_array.items.len;
        if (i > 0 and remaining > 0) {
            cumulative_count.* -= @popCount(bit_array.items[i - 1]);
            const new = (bit_array.items[i - 1] >> @intCast(8 - remaining)) << @intCast(8 - remaining);
            cumulative_count.* += @popCount(new);
            bit_array.items[i - 1] = new;
        }
        hl_rank.appendAssumeCapacity(cumulative_count.*);
    }

    fn build_rank(
        allocator: std.mem.Allocator,
        bits: []const u8,
        num_bits: u64,
    ) !struct { hl_rank: []u64, bit_array: []align(64) u8, count: u64 } {
        var hl_rank, var bit_array = try alloc_rank_arrays(allocator, num_bits);
        defer hl_rank.deinit(allocator);
        defer bit_array.deinit(allocator);

        var cumulative_count: u64 = 0;
        var relative_count: u16 = 0;
        var k: usize = 0;
        while ((k << 3) < num_bits) {
            rank_byte(bits[k], k, &cumulative_count, &relative_count, &hl_rank, &bit_array);
            k += 1;
        }

        rank_last_byte(@intCast(num_bits & 0b111), &cumulative_count, &hl_rank, &bit_array);

        return .{
            .hl_rank = try hl_rank.toOwnedSlice(allocator),
            .bit_array = try bit_array.toOwnedSlice(allocator),
            .count = cumulative_count,
        };
    }

    fn build_select(
        allocator: std.mem.Allocator,
        cumulative_count: u64,
        num_bits: u64,
        bit_array: []align(64) u8,
    ) !struct { hl_select: []u64, ll_select: []u16, sigma_hl_pow: u6, sigma_ll_pow: u4 } {
        var hl_select = std.ArrayListUnmanaged(u64){};
        defer hl_select.deinit(allocator);
        var ll_select = std.ArrayListUnmanaged(u16){};
        defer ll_select.deinit(allocator);

        const s_hl_pow: u6 = @intFromFloat(@ceil(@log2(@as(f64, @floatFromInt(sb_size * cumulative_count)) / @as(f64, @floatFromInt(num_bits)))));
        const s_ll_pow: u4 = @intFromFloat(@ceil(@log2(4096.0 * 0.99 * @as(f64, @floatFromInt(cumulative_count)) / @as(f64, @floatFromInt(num_bits)))));

        try hl_select.ensureTotalCapacityPrecise(allocator, 2 + (cumulative_count >> s_hl_pow));
        try ll_select.ensureTotalCapacityPrecise(allocator, 2 + (cumulative_count >> s_ll_pow));

        var pos: u64 = 0;
        var k: u64 = 0;

        for (0..bit_array.len) |m| {
            if (m % 64 == 0 or (m - 1) % 64 == 0) continue;
            if (bit_array[m] == 0) continue;
            pos = ((m - ((m >> 6) + 1) * 2) << 3) + @clz(bit_array[m]);
            k = m;
            break;
        }

        hl_select.appendAssumeCapacity((pos + (sb_size >> 1)) / sb_size);
        ll_select.appendAssumeCapacity(@intCast(pos - ((pos >> s_ll_pow) << s_ll_pow)));

        var ll_count: u64 = 0;
        var hl_count: u64 = 0;

        var sel_n1: u64 = 0;
        for (k..bit_array.len) |m| {
            if (m % 64 == 0 or (m - 1) % 64 == 0) continue;
            pos = (m - ((m >> 6) + 1) * 2) << 3;

            const byte = bit_array[m];
            const count = @popCount(byte);

            if (hl_count + count < (@as(u64, 1) << s_hl_pow)) {
                hl_count += count;
            } else {
                const ones_left = (@as(u64, 1) << s_hl_pow) - hl_count;
                const r_pos = try bitops.nthSetBitPos(byte, @intCast(ones_left));
                hl_select.appendAssumeCapacity(((pos + r_pos) + (sb_size >> 1)) / sb_size);
                hl_count = count - @popCount(@as(u16, bit_array[m]) >> @intCast(ones_left));
            }

            if (ll_count + count < (@as(u16, 1) << s_ll_pow)) {
                ll_count += count;
            } else {
                const ones_left = (@as(u16, 1) << s_ll_pow) - ll_count;
                const r_pos = try bitops.nthSetBitPos(byte, @intCast(ones_left));
                ll_select.appendAssumeCapacity(@intCast((pos + r_pos) - ((pos >> s_ll_pow) << s_ll_pow)));
                ll_count = count - @popCount(@as(u16, bit_array[m]) >> @intCast(ones_left));
            }

            if (count > 0) {
                sel_n1 = pos + try bitops.nthSetBitPos(byte, @intCast(count));
            }
        }

        hl_select.appendAssumeCapacity(hl_select.capacity - 1);
        ll_select.appendAssumeCapacity(@intCast(sel_n1 - sb_size * (sel_n1 / sb_size)));

        return .{
            .hl_select = try hl_select.toOwnedSlice(allocator),
            .ll_select = try ll_select.toOwnedSlice(allocator),
            .sigma_hl_pow = s_hl_pow,
            .sigma_ll_pow = s_ll_pow,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.hl_rank);
        allocator.free(self.hl_select);
        allocator.free(self.ll_select);
        allocator.free(self.bit_array);
        self.* = undefined;
    }

    pub fn rank(self: *Self, i: u64) Error!u64 {
        return self.rank1b1(i + 1);
    }

    pub fn select(self: *Self, i: u64) Error!u64 {
        return self.select1b1(i + 1);
    }

    pub fn rankB1(self: *Self, i: u64) Error!u64 {
        if (i == 0) return Error.Undefined;
        if (i > self.size) return Error.OutOfBounds;

        const x = i / 496;
        const n_bits = i - (x * 496);
        const ri = x >> 7;

        var count: u16 = 0;
        if (n_bits > 7) {
            for (self.bit_array[(x << 6) + 2 ..][0..(n_bits >> 3)]) |byte| {
                count += @popCount(byte);
            }
        }

        const remaining = n_bits & 0b111;
        if (remaining > 0) {
            count += @popCount(self.bit_array[(x << 6) + 2 + (n_bits >> 3)] >> @intCast(8 - remaining));
        }

        return self.hl_rank[ri] + self.r_count(x) + count;
    }

    pub fn selectB1(self: *Self, i: u64) Error!u64 {
        if (i == 0) return Error.Undefined;
        if (i > self.hl_rank[self.hl_rank.len - 1]) return Error.OutOfBounds;

        var s = (i - 1) >> self.sigma_hl_pow;
        // while (i <= self.hl_rank[s] or i > self.hl_rank[s + 1]) {
        //     if (i <= self.hl_rank[s]) s += 1 else s -|= 1;
        // }

        while (i <= self.hl_rank[s]) s -|= 1;
        while (i > self.hl_rank[s + 1]) s +|= 1;

        const l = (i - 1) >> self.sigma_ll_pow;
        var a: u64 = @intCast(self.ll_select[l]);
        var b: u64 = @intCast(self.ll_select[l + 1]);

        if (b < a) {
            if (self.hl_rank[s] >= (l << self.sigma_ll_pow)) {
                a -|= sb_size;
            } else {
                b +|= sb_size;
            }
        }

        var p = @as(u64, s << 10) + (a + ((i - (l << self.sigma_ll_pow)) * (b - a) >> self.sigma_ll_pow)) / 62;
        // var p: u64 = (sb_size * s + a + (((i - (l << self.sigmaLlPow)) * (b - a)) >> self.sigmaLlPow)) / 496;

        while ((p << 6 >= self.bit_array.len) or i <= self.r_count(p) + self.hl_rank[p >> 7]) p -|= 1;
        while (((p + 1) << 6 < self.bit_array.len) and i > self.r_count(p + 1) + self.hl_rank[(p + 1) >> 7]) p +|= 1;

        const target = i - (self.r_count(p) + self.hl_rank[p >> 7]);

        var rel_count: usize = 0;
        for (((p << 6) + 2)..self.bit_array.len) |m| {
            if (m % 64 == 0 or (m - 1) % 64 == 0) continue;

            const byte = self.bit_array[m];
            const count = @popCount(byte);

            if (rel_count + count < target) {
                rel_count += count;
            } else {
                const r_pos = bitops.nthSetBitPos(byte, @intCast(target - rel_count)) catch 0;
                return ((m - ((m >> 6) + 1) * 2) << 3) + r_pos;
            }
        }

        return Error.Undefined;
    }

    fn r_count(self: *Self, i: usize) u16 {
        return std.mem.readInt(u16, self.bit_array[(i << 6)..][0..2], builtin.cpu.arch.endian());
    }

    const Builder = struct {
        pub const Error = error{InvalidN};
        const bufferSize = 128;
        const Self = @This();

        hl_rank: std.ArrayListUnmanaged(u64),
        bit_array: std.ArrayListAlignedUnmanaged(u8, 64),
        num_bits: u64,
        bits_written: u64,
        cumulative_count: u64,
        relative_count: u16,
        buffer: u8,

        pub fn initWithTotalCapacity(allocator: std.mem.Allocator, num_bits: u64) !Builder.Self {
            const hl_rank, const bit_array = try alloc_rank_arrays(allocator, num_bits);

            return Builder.Self{
                .hl_rank = hl_rank,
                .bit_array = bit_array,
                .buffer = 0,
                .bits_written = 0,
                .num_bits = num_bits,
                .cumulative_count = 0,
                .relative_count = 0,
            };
        }

        fn init() !Builder.Self {}

        pub fn deinit(self: *Builder.Self, allocator: std.mem.Allocator) void {
            self.hl_rank.deinit(allocator);
            self.bit_array.deinit(allocator);
            self.* = undefined;
        }

        pub fn append(self: *Builder.Self, data: anytype, n: u9) Builder.Error!void {
            const T = @TypeOf(data);
            if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned)
                @compileError("Spider.Builder.add requires an unsigned integer, found " ++ @typeName(T));

            if (n == 0 or n > @typeInfo(T).int.bits) {
                return Builder.Error.InvalidN;
            }

            var remaining = n;
            while (remaining > 0) {
                const buffer_pos = self.bits_written & 0b111;
                const to_write = @min(@as(u4, 8) - buffer_pos, remaining);
                const mask = (@as(u16, 1) << @intCast(to_write)) - 1;
                self.buffer |= @truncate((data >> @intCast(remaining - to_write)) & mask);

                if (buffer_pos + to_write >= 8) {
                    rank_byte(self.buffer, self.bits_written >> 3, &self.cumulative_count, &self.relative_count, &self.hl_rank, &self.bit_array);
                    self.buffer = 0;
                }

                self.bits_written += to_write;
                remaining -= to_write;
            }
        }

        pub fn build(self: *Builder.Self, allocator: std.mem.Allocator) !Spider {
            Spider.rank_last_byte(@intCast(self.num_bits & 0b111), &self.cumulative_count, &self.hl_rank, &self.bit_array);

            const bit_arr = try self.bit_array.toOwnedSlice(allocator);

            const sel = try build_select(allocator, self.cumulative_count, self.num_bits, bit_arr);

            return Spider{
                .hl_rank = try self.hl_rank.toOwnedSlice(allocator),
                .bit_array = bit_arr,
                .hl_select = sel.hl_select,
                .ll_select = sel.ll_select,
                .sigma_hl_pow = sel.sigma_hl_pow,
                .sigma_ll_pow = sel.sigma_ll_pow,

                .size = self.num_bits,
            };
        }
    };
};

test "spiderBuilder" {
    const seed = 0x4C471781B1;
    const buf_size = 1 << 10;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try std.testing.allocator.alloc(u8, buf_size);
    defer std.testing.allocator.free(data);

    data.len = buf_size;
    random.bytes(data);

    var builder = try Spider.Builder.initWithTotalCapacity(std.testing.allocator, buf_size * 8);
    defer builder.deinit(std.testing.allocator);

    for (data) |x| {
        try builder.append(x, 8);
    }

    var spider = try builder.build(std.testing.allocator);
    defer spider.deinit(std.testing.allocator);

    var count: u64 = 0;

    for (0..buf_size) |i| {
        const bit = random.uintAtMost(u4, 8);
        const expectation = count + @popCount(@as(u16, data[i]) >> @intCast(8 - bit));
        const rank = try spider.rankB1((i * 8) + bit);
        try std.testing.expectEqual(expectation, rank);
        count += @popCount(data[i]);
    }

    try std.testing.expectEqual(count, try spider.rankB1(buf_size * 8));
    try std.testing.expectError(Spider.Error.Undefined, spider.rankB1(0));
}

test "spiderRank" {
    const seed = 0x71CD3A1E15;
    const buf_size = 1 << 20;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try std.testing.allocator.alloc(u8, buf_size);
    defer std.testing.allocator.free(data);

    data.len = buf_size;
    random.bytes(data);

    var spider = try Spider.init(std.testing.allocator, data, buf_size * 8);
    defer spider.deinit(std.testing.allocator);

    var count: u64 = 0;

    for (0..buf_size) |i| {
        const bit = random.uintAtMost(u4, 8);
        const expectation = count + @popCount(@as(u16, data[i]) >> @intCast(8 - bit));
        const rank = try spider.rankB1((i * 8) + bit);
        try std.testing.expectEqual(expectation, rank);
        count += @popCount(data[i]);
    }

    try std.testing.expectEqual(count, try spider.rankB1(buf_size * 8));
    try std.testing.expectError(Spider.Error.Undefined, spider.rankB1(0));
}

test "spiderSelect" {
    const seed = 0x4C27A681B1;
    const buf_size = 1 << 10;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try std.testing.allocator.alloc(u8, buf_size);
    defer std.testing.allocator.free(data);

    data.len = buf_size;
    random.bytes(data);

    var spider = try Spider.init(std.testing.allocator, data, buf_size * 8);
    defer spider.deinit(std.testing.allocator);

    var count: u64 = 0;

    for (0..buf_size) |i| {
        if (data[i] == 0) continue;
        const n = random.uintAtMost(u4, @popCount(data[i]) - 1) + 1;
        const pos = try bitops.nthSetBitPos(data[i], n);
        const expectation = i * 8 + pos;

        const select = try spider.selectB1(count + n);
        try std.testing.expectEqual(expectation, select);

        count += @popCount(data[i]);
    }

    try std.testing.expectError(Spider.Error.Undefined, spider.selectB1(0));
    try std.testing.expectEqual(2, spider.selectB1(1));
}
