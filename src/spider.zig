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

    sigmaHlPow: u6,
    sigmaLlPow: u4,

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
            .sigmaHlPow = sel.sigmaHlPow,
            .sigmaLlPow = sel.sigmaLlPow,

            .size = num_bits,
        };
    }

    fn build_rank(
        allocator: std.mem.Allocator,
        bits: []const u8,
        num_bits: u64,
    ) !struct { hl_rank: []u64, bit_array: []align(64) u8, count: u64 } {
        const nHl = (num_bits - 1) / sb_size + 2;
        const nRank = ((num_bits - 1) / (sb_size >> 7) + 1) * 64 + 2;

        var hl_rank = std.ArrayListUnmanaged(u64){};
        defer hl_rank.deinit(allocator);

        var bit_array = std.ArrayListAlignedUnmanaged(u8, 64){};
        defer bit_array.deinit(allocator);

        try hl_rank.ensureTotalCapacityPrecise(allocator, nHl);
        try bit_array.ensureTotalCapacityPrecise(allocator, nRank);
        bit_array.items.len = nRank;

        var cumulative_count: u64 = 0;
        var relative_count: u16 = 0;

        var i: usize = 0;

        var k: usize = 0;
        while ((k << 3) < num_bits) {
            if (k % (sb_size >> 3) == 0) { // k % 7936
                hl_rank.appendAssumeCapacity(cumulative_count);
                relative_count = 0;
            }

            if (k % (sb_size >> 10) == 0) { // k % 62
                std.mem.writeInt(u16, bit_array.items[i..][0..2], relative_count, builtin.cpu.arch.endian());
                i += 2;
            }

            cumulative_count += @popCount(bits[k]);
            relative_count += @popCount(bits[k]);

            bit_array.items[i] = bits[k];
            i += 1;
            k += 1;
        }

        const remaining = num_bits & 0b111;
        if (i > 0 and remaining > 0) {
            cumulative_count -= @popCount(bit_array.items[i - 1]);
            const new = (bit_array.items[i - 1] >> @intCast(8 - remaining)) << @intCast(8 - remaining);
            cumulative_count += @popCount(new);
            bit_array.items[i - 1] = new;
        }

        hl_rank.appendAssumeCapacity(cumulative_count);
        bit_array.items.len = i;

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
    ) !struct { hl_select: []u64, ll_select: []u16, sigmaHlPow: u6, sigmaLlPow: u4 } {
        var hl_select = std.ArrayListUnmanaged(u64){};
        defer hl_select.deinit(allocator);
        var ll_select = std.ArrayListUnmanaged(u16){};
        defer ll_select.deinit(allocator);

        const sigmaHlPow: u6 = @intFromFloat(@ceil(@log2(@as(f64, @floatFromInt(sb_size * cumulative_count)) / @as(f64, @floatFromInt(num_bits)))));
        const sigmaLlPow: u4 = @intFromFloat(@ceil(@log2(4096.0 * 0.99 * @as(f64, @floatFromInt(cumulative_count)) / @as(f64, @floatFromInt(num_bits)))));

        try hl_select.ensureTotalCapacityPrecise(allocator, 2 + (cumulative_count >> sigmaHlPow));
        try ll_select.ensureTotalCapacityPrecise(allocator, 2 + (cumulative_count >> sigmaLlPow));

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
        ll_select.appendAssumeCapacity(@intCast(pos - ((pos >> sigmaLlPow) << sigmaLlPow)));

        var ll_count: u64 = 0;
        var hl_count: u64 = 0;

        var sel_n1: u64 = 0;
        for (k..bit_array.len) |m| {
            if (m % 64 == 0 or (m - 1) % 64 == 0) continue;
            pos = (m - ((m >> 6) + 1) * 2) << 3;

            const byte = bit_array[m];
            const count = @popCount(byte);

            if (hl_count + count < (@as(u64, 1) << sigmaHlPow)) {
                hl_count += count;
            } else {
                const ones_left = (@as(u64, 1) << sigmaHlPow) - hl_count;
                const r_pos = try bitops.nthSetBitPos(byte, @intCast(ones_left));
                hl_select.appendAssumeCapacity(((pos + r_pos) + (sb_size >> 1)) / sb_size);
                hl_count = count - @popCount(@as(u16, bit_array[m]) >> @intCast(ones_left));
            }

            if (ll_count + count < (@as(u16, 1) << sigmaLlPow)) {
                ll_count += count;
            } else {
                const ones_left = (@as(u16, 1) << sigmaLlPow) - ll_count;
                const r_pos = try bitops.nthSetBitPos(byte, @intCast(ones_left));
                ll_select.appendAssumeCapacity(@intCast((pos + r_pos) - ((pos >> sigmaLlPow) << sigmaLlPow)));
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
            .sigmaHlPow = sigmaHlPow,
            .sigmaLlPow = sigmaLlPow,
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
        const nBits = i - (x * 496);
        const ri = x >> 7;

        var count: u16 = 0;
        if (nBits > 7) {
            for (self.bit_array[(x << 6) + 2 ..][0..(nBits >> 3)]) |byte| {
                count += @popCount(byte);
            }
        }

        const remaining = nBits & 0b111;
        if (remaining > 0) {
            count += @popCount(self.bit_array[(x << 6) + 2 + (nBits >> 3)] >> @intCast(8 - remaining));
        }

        return self.hl_rank[ri] + self.r_count(x) + count;
    }

    pub fn selectB1(self: *Self, i: u64) Error!u64 {
        if (i == 0) return Error.Undefined;
        if (i > self.hl_rank[self.hl_rank.len - 1]) return Error.OutOfBounds;

        var s = (i - 1) >> self.sigmaHlPow;
        // while (i <= self.hl_rank[s] or i > self.hl_rank[s + 1]) {
        //     if (i <= self.hl_rank[s]) s += 1 else s -|= 1;
        // }

        while (i <= self.hl_rank[s]) s -|= 1;
        while (i > self.hl_rank[s + 1]) s +|= 1;

        const l = (i - 1) >> self.sigmaLlPow;
        var a: u64 = @intCast(self.ll_select[l]);
        var b: u64 = @intCast(self.ll_select[l + 1]);

        if (b < a) {
            if (self.hl_rank[s] >= (l << self.sigmaLlPow)) {
                a -|= sb_size;
            } else {
                b +|= sb_size;
            }
        }

        var p = @as(u64, s << 10) + (a + ((i - (l << self.sigmaLlPow)) * (b - a) >> self.sigmaLlPow)) / 62;
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
};

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
