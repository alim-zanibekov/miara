const std = @import("std");
const builtin = @import("builtin");
const util = @import("util.zig");
const bit = @import("bit.zig");

/// Data structure for bit-based rank and select queries
pub const Spider = struct {
    pub const Error = error{OutOfBounds} || std.mem.Allocator.Error || bit.NthSetBitError;

    const sb_size = 63488; // superblock_size
    const Self = @This();

    const SelectData = struct {
        hl_select: []const u64,
        ll_select: []const u16,
        sigma_hl_pow: u6,
        sigma_ll_pow: u4,

        fn byteSize(self: *const SelectData) u64 {
            return self.hl_select.len * @sizeOf(u64) + self.ll_select.len * @sizeOf(u16) + @sizeOf(Self);
        }
    };

    bit_array: []align(64) const u8,
    hl_rank: []const u64,
    size: u64,
    sd1: SelectData,
    sd0: SelectData,

    /// Creates a SPIDER from a bit array and total number of bits
    pub fn init(allocator: std.mem.Allocator, bits: []const u8, num_bits: u64) Error!Self {
        if (num_bits == 0) return Error.OutOfBounds;
        if ((num_bits - 1) >> 3 + 1 > bits.len) return Error.OutOfBounds;

        const rnk = try build_rank(allocator, bits, num_bits);
        const sel1 = try build_select(allocator, rnk.count, num_bits, rnk.bit_array, false);
        const sel0 = try build_select(allocator, rnk.count, num_bits, rnk.bit_array, true);

        return .{
            .hl_rank = rnk.hl_rank,
            .bit_array = rnk.bit_array,
            .sd1 = sel1,
            .sd0 = sel0,
            .size = num_bits,
        };
    }

    fn alloc_rank_arrays(allocator: std.mem.Allocator, num_bits: u64) std.mem.Allocator.Error!struct {
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
    ) Error!struct { hl_rank: []u64, bit_array: []align(64) u8, count: u64 } {
        var hl_rank, var bit_array = try alloc_rank_arrays(allocator, num_bits);
        defer hl_rank.deinit(allocator);
        defer bit_array.deinit(allocator);

        var cumulative_count: u64 = 0;
        var relative_count: u16 = 0;
        var k: usize = 0;
        while ((k << 3) < num_bits) : (k += 1) {
            rank_byte(bits[k], k, &cumulative_count, &relative_count, &hl_rank, &bit_array);
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
        comptime flip: bool,
    ) Error!SelectData {
        var hl_select = std.ArrayListUnmanaged(u64){};
        defer hl_select.deinit(allocator);
        var ll_select = std.ArrayListUnmanaged(u16){};
        defer ll_select.deinit(allocator);

        const num_bytes = bit_array.len;
        const c_count = if (flip) num_bits - cumulative_count else cumulative_count;

        const s_hl_pow: u6 = @intCast(util.log2IntCeilOrZero(u64, @intFromFloat(@ceil(toF64(sb_size * c_count) / toF64(num_bits)))));
        const s_ll_pow: u4 = @intCast(util.log2IntCeilOrZero(u64, @intFromFloat(@ceil(4096.0 * 0.99 * toF64(c_count) / toF64(num_bits)))));

        try hl_select.ensureTotalCapacityPrecise(allocator, @as(usize, 3) + (std.math.divCeil(usize, c_count, @as(u64, 1) << s_hl_pow) catch 0));
        try ll_select.ensureTotalCapacityPrecise(allocator, @as(usize, 3) + (std.math.divCeil(usize, c_count, @as(u64, 1) << s_ll_pow) catch 0));
        var pos: u64 = 0;
        var k: u64 = 0;

        for (0..num_bytes) |m| {
            if (m % 64 == 0 or (m - 1) % 64 == 0) continue;
            const it = if (flip) ~bit_array[m] else bit_array[m];
            if (it == 0) continue;
            pos = ((m - ((m >> 6) + 1) * 2) << 3) + @clz(it);
            k = m;
            break;
        }

        hl_select.appendAssumeCapacity((pos + (sb_size >> 1)) / sb_size);
        ll_select.appendAssumeCapacity(@intCast(pos - ((pos >> s_ll_pow) << s_ll_pow)));

        var ll_count: u64 = 0;
        var hl_count: u64 = 0;

        var sel_n1: u64 = 0;

        for (k..num_bytes) |m| {
            if (m % 64 == 0 or (m - 1) % 64 == 0) continue;
            pos = (m - ((m >> 6) + 1) * 2) << 3;

            const byte = if (flip) ~bit_array[m] else bit_array[m];
            const count = @popCount(byte);

            if (hl_count + count < (@as(u64, 1) << s_hl_pow)) {
                hl_count += count;
            } else {
                const ones_left = (@as(u64, 1) << s_hl_pow) - hl_count;
                const r_pos = try bit.nthSetBitPos(byte, @intCast(ones_left));
                hl_select.appendAssumeCapacity(((pos + r_pos) + (sb_size >> 1)) / sb_size);
                hl_count = count - ones_left;
            }

            if (ll_count + count < (@as(u16, 1) << s_ll_pow)) {
                ll_count += count;
            } else {
                const ones_left = (@as(u16, 1) << s_ll_pow) - ll_count;
                const r_pos = try bit.nthSetBitPos(byte, @intCast(ones_left));
                ll_select.appendAssumeCapacity(@intCast((pos + r_pos) - ((pos >> s_ll_pow) << s_ll_pow)));
                ll_count = count - ones_left;
            }

            if (count > 0) {
                sel_n1 = pos + try bit.nthSetBitPos(byte, @intCast(count));
            }
        }

        hl_select.appendAssumeCapacity(hl_select.capacity - 1);
        ll_select.appendAssumeCapacity(@intCast(sel_n1 - sb_size * (sel_n1 / sb_size)));

        return SelectData{
            .hl_select = try hl_select.toOwnedSlice(allocator),
            .ll_select = try ll_select.toOwnedSlice(allocator),
            .sigma_hl_pow = s_hl_pow,
            .sigma_ll_pow = s_ll_pow,
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.hl_rank);
        allocator.free(self.sd1.hl_select);
        allocator.free(self.sd0.hl_select);
        allocator.free(self.sd1.ll_select);
        allocator.free(self.sd0.ll_select);
        allocator.free(self.bit_array);
        self.* = undefined;
    }

    /// Returns the number of 1 bits up to and including index `i`
    pub fn rank1(self: *Self, i: u64) Error!u64 {
        if (i == 0 or i > self.size) return Error.OutOfBounds;

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

        return self.hl_rank[ri] + self.r_count1(x) + count;
    }

    /// Returns the position of the i-th 0 bit. `i` - 1-based index; result - 0-based index
    pub fn select0(self: *const Self, i: u64) Error!u64 {
        return self.select_general(i, &self.sd0, true);
    }

    /// Returns the position of the i-th 1 bit. `i` - 1-based index; result - 0-based index
    pub fn select1(self: *const Self, i: u64) Error!u64 {
        return self.select_general(i, &self.sd1, false);
    }

    fn select_general(self: *const Self, i: u64, s_data: *const SelectData, comptime flip: bool) Error!u64 {
        const count = if (flip) Spider.count0 else Spider.count1;
        const r_count = if (flip) Spider.r_count0 else Spider.r_count1;

        if (i == 0 or i > count(self, self.hl_rank.len - 1)) return Error.OutOfBounds;

        var s = (i - 1) >> s_data.sigma_hl_pow;
        // while (i <= self.hl_rank[s] or i > self.hl_rank[s + 1]) {
        //     if (i <= self.hl_rank[s]) s += 1 else s -|= 1;
        // }

        while (i <= count(self, s)) s -|= 1;
        while (i > count(self, s + 1)) s +|= 1;

        const l = (i - 1) >> s_data.sigma_ll_pow;
        var a: u64 = @intCast(s_data.ll_select[l]);
        var b: u64 = @intCast(s_data.ll_select[l + 1]);

        if (b < a) {
            if (count(self, s) >= (l << s_data.sigma_ll_pow)) {
                a -|= sb_size;
            } else {
                b +|= sb_size;
            }
        }

        var p = @as(u64, s << 10) + (a + ((i - (l << s_data.sigma_ll_pow)) * (b - a) >> s_data.sigma_ll_pow)) / 62;
        // var p: u64 = (sb_size * s + a + (((i - (l << self.sigmaLlPow)) * (b - a)) >> self.sigmaLlPow)) / 496;

        const num_bytes = self.bit_array.len;

        while ((p << 6 >= num_bytes) or i <= r_count(self, p) + count(self, p >> 7)) p -|= 1;
        while (((p + 1) << 6 < num_bytes) and i > r_count(self, p + 1) + count(self, (p + 1) >> 7)) p +|= 1;

        const target = i - (r_count(self, p) + count(self, p >> 7));

        var rel_count: usize = 0;
        for (((p << 6) + 2)..num_bytes) |m| {
            if (m % 64 == 0 or (m - 1) % 64 == 0) continue;

            const byte = if (flip) ~self.bit_array[m] else self.bit_array[m];
            const b_count = @popCount(byte);

            if (rel_count + b_count < target) {
                rel_count += b_count;
            } else {
                const r_pos = bit.nthSetBitPos(byte, @intCast(target - rel_count)) catch 0;
                return ((m - ((m >> 6) + 1) * 2) << 3) + r_pos;
            }
        }

        return Error.NotFound;
    }

    fn count0(self: *const Self, i: usize) u64 {
        return (i * sb_size) - self.hl_rank[i];
    }

    fn count1(self: *const Self, i: usize) u64 {
        return self.hl_rank[i];
    }

    fn r_count0(self: *const Self, i: usize) u16 {
        const c1 = std.mem.readInt(u16, self.bit_array[(i << 6)..][0..2], builtin.cpu.arch.endian());
        const start: u16 = @intCast(i * (sb_size >> 7) % sb_size);
        return start - c1;
    }

    fn r_count1(self: *const Self, i: usize) u16 {
        return std.mem.readInt(u16, self.bit_array[(i << 6)..][0..2], builtin.cpu.arch.endian());
    }

    /// Builder for constructing SPIDER
    pub const Builder = struct {
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

        /// Creates a Builder that can hold up to `num_bits` bits
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

        pub fn deinit(self: *Builder.Self, allocator: std.mem.Allocator) void {
            self.hl_rank.deinit(allocator);
            self.bit_array.deinit(allocator);
            self.* = undefined;
        }

        /// Appends `n` lsb of `data` to the bitstream
        pub fn append(self: *Builder.Self, data: anytype, n: u16) Builder.Error!void {
            const T = @TypeOf(data);
            const U = u8;
            if (@typeInfo(T) != .int or @typeInfo(T).int.signedness != .unsigned)
                @compileError("Spider.Builder.add requires an unsigned integer, found " ++ @typeName(T));

            if (n == 0 or n > @typeInfo(T).int.bits) {
                return Builder.Error.InvalidN;
            }
            var remaining = n;
            while (remaining > 0) {
                const buffer_pos: std.math.Log2IntCeil(U) = @intCast(self.bits_written % @bitSizeOf(U));
                const to_write: std.math.Log2IntCeil(U) = @intCast(@min(@bitSizeOf(U) - buffer_pos, remaining));
                const mask = ~@as(U, 0) >> @intCast(@as(u16, @bitSizeOf(U)) - to_write);
                const value = @as(U, @intCast((data >> @intCast(remaining - to_write) & mask)));
                const shift: u16 = @as(u16, @bitSizeOf(U)) - to_write - buffer_pos;
                self.buffer |= @intCast(value << @intCast(shift));
                if (buffer_pos + to_write >= @bitSizeOf(U)) {
                    rank_byte(self.buffer, self.bits_written >> 3, &self.cumulative_count, &self.relative_count, &self.hl_rank, &self.bit_array);
                    self.buffer = 0;
                }
                self.bits_written += to_write;
                remaining -= to_write;
            }
        }

        /// Finalizes and builds SPIDER
        pub fn build(self: *Builder.Self, allocator: std.mem.Allocator) !Spider {
            if (self.buffer != 0) {
                rank_byte(self.buffer, self.bits_written >> 3, &self.cumulative_count, &self.relative_count, &self.hl_rank, &self.bit_array);
            }
            Spider.rank_last_byte(@intCast(self.num_bits & 0b111), &self.cumulative_count, &self.hl_rank, &self.bit_array);

            const bit_arr = try self.bit_array.toOwnedSlice(allocator);

            const sel1 = try build_select(allocator, self.cumulative_count, self.num_bits, bit_arr, false);
            const sel0 = try build_select(allocator, self.cumulative_count, self.num_bits, bit_arr, true);

            return Spider{
                .hl_rank = try self.hl_rank.toOwnedSlice(allocator),
                .bit_array = bit_arr,
                .sd1 = sel1,
                .sd0 = sel0,
                .size = self.bits_written,
            };
        }
    };
};

const toF64 = util.toF64;

const testing = std.testing;

test "spiderBuilder" {
    const seed = 0x4C471781B1;
    const buf_size = 1 << 10;

    const U = usize;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try testing.allocator.alloc(U, buf_size);
    defer testing.allocator.free(data);
    data.len = buf_size;
    for (0..buf_size) |i| {
        data[i] = random.int(U);
    }

    var builder = try Spider.Builder.initWithTotalCapacity(testing.allocator, buf_size * @bitSizeOf(U));
    defer builder.deinit(testing.allocator);

    for (data) |x| {
        const toAdd = random.uintAtMost(u16, @bitSizeOf(U) - 1) + 1;
        const remaining = @bitSizeOf(U) - toAdd;
        try builder.append(x >> @intCast(remaining), toAdd);
        if (remaining != 0) {
            try builder.append(x, remaining);
        }
    }

    var spider = try builder.build(testing.allocator);
    defer spider.deinit(testing.allocator);

    var count: u64 = 0;

    for (0..buf_size) |i| {
        const b = random.uintAtMost(U, @bitSizeOf(U) - 1) + 1;
        const expectation = count + @popCount(data[i] >> @intCast(@bitSizeOf(U) - b));
        const rank = try spider.rank1((i * @bitSizeOf(U)) + b);
        try testing.expectEqual(expectation, rank);
        count += @popCount(data[i]);
    }

    try testing.expectEqual(count, try spider.rank1(buf_size * @bitSizeOf(U)));
    try testing.expectError(Spider.Error.OutOfBounds, spider.rank1(0));
}

test "spiderRank" {
    const seed = 0x71CD3A1E15;
    const buf_size = 1 << 20;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try testing.allocator.alloc(u8, buf_size);
    defer testing.allocator.free(data);

    data.len = buf_size;
    random.bytes(data);

    var spider = try Spider.init(testing.allocator, data, buf_size * 8);
    defer spider.deinit(testing.allocator);

    var count: u64 = 0;

    for (0..buf_size) |i| {
        const b = random.uintAtMost(u4, 8);
        const expectation = count + @popCount(@as(u16, data[i]) >> @intCast(8 - b));
        const rank = try spider.rank1((i * 8) + b);
        try testing.expectEqual(expectation, rank);
        count += @popCount(data[i]);
    }

    try testing.expectEqual(count, try spider.rank1(buf_size * 8));
    try testing.expectError(Spider.Error.OutOfBounds, spider.rank1(0));
}

test "spiderSelect1" {
    const seed = 0x4C27A681B1;
    const buf_size = 1 << 10;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try testing.allocator.alloc(u8, buf_size);
    defer testing.allocator.free(data);

    data.len = buf_size;
    random.bytes(data);

    var spider = try Spider.init(testing.allocator, data, buf_size * 8);
    defer spider.deinit(testing.allocator);

    var count: u64 = 0;

    for (0..buf_size) |i| {
        if (data[i] == 0) continue;
        const n = random.uintAtMost(u4, @popCount(data[i]) - 1) + 1;
        const pos = try bit.nthSetBitPos(data[i], n);
        const expectation = i * 8 + pos;

        const select = try spider.select1(count + n);
        try testing.expectEqual(expectation, select);

        count += @popCount(data[i]);
    }

    try testing.expectError(Spider.Error.OutOfBounds, spider.select1(0));
    try testing.expectEqual(2, spider.select1(1));
}

test "spiderSelect0" {
    const seed = 0x3D93F7A0B1;
    const buf_size = 1 << 10;

    var r = std.Random.DefaultPrng.init(seed);
    var random = r.random();

    var data = try testing.allocator.alloc(u8, buf_size);
    defer testing.allocator.free(data);

    data.len = buf_size;
    random.bytes(data);

    var spider = try Spider.init(testing.allocator, data, buf_size * 8);
    defer spider.deinit(testing.allocator);

    var count: u64 = 0;

    for (0..buf_size) |i| {
        const byte = ~data[i];
        if (byte == 0) continue;
        const n = random.uintAtMost(u4, @popCount(byte) - 1) + 1;
        const pos = try bit.nthSetBitPos(byte, n);
        const expectation = i * 8 + pos;

        const select = try spider.select0(count + n);
        try testing.expectEqual(expectation, select);

        count += @popCount(byte);
    }

    try testing.expectError(Spider.Error.OutOfBounds, spider.select1(0));
    try testing.expectEqual(2, spider.select1(1));
}
