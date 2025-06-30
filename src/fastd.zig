const std = @import("std");

pub const MagicU32 = struct { u64 };

pub const MagicU64 = struct { u128 };

pub inline fn fastdiv(m: anytype, n: ReverseMagicType(@TypeOf(m))) ReverseMagicType(@TypeOf(m)) {
    return switch (@TypeOf(m)) {
        MagicU32 => fastdiv_u32(n, m.@"0"),
        MagicU64 => fastdiv_u64(n, m.@"0"),
        else => @compileError("Unsupported magic type " ++ @typeName(@TypeOf(m))),
    };
}

pub inline fn fastmod(m: anytype, n: ReverseMagicType(@TypeOf(m)), d: ReverseMagicType(@TypeOf(m))) ReverseMagicType(@TypeOf(m)) {
    return switch (@TypeOf(m)) {
        MagicU32 => fastmod_u32(n, m.@"0", d),
        MagicU64 => fastmod_u64(n, m.@"0", d),
        else => @compileError("Unsupported magic type " ++ @typeName(@TypeOf(m))),
    };
}

pub inline fn is_divisible(m: anytype, n: ReverseMagicType(@TypeOf(m))) bool {
    return switch (@TypeOf(m)) {
        MagicU32 => is_divisible_u32(n, m.@"0"),
        MagicU64 => is_divisible_u64(n, m.@"0"),
        else => @compileError("Unsupported magic type " ++ @typeName(@TypeOf(m))),
    };
}

pub inline fn magicNumber(divisor: anytype) MagicType(@TypeOf(divisor)) {
    const T = MagicType(@TypeOf(divisor));
    return switch (T) {
        MagicU32 => .{magic_u32(divisor)},
        MagicU64 => .{magic_u64(divisor)},
        else => @compileError("Unsupported magic number type " ++ @typeName(T)),
    };
}

const u32_max = ~@as(u32, 0);
const u64_max = ~@as(u64, 0);
const u128_max = ~@as(u128, 0);

pub inline fn fastdiv_u32(n: u32, m: u64) u32 {
    return @intCast(mul128_u32(n, m));
}

pub inline fn fastdiv_u64(n: u64, m: u128) u64 {
    return mul128_u64(n, m);
}

pub inline fn fastmod_u32(n: u32, m: u64, d: u32) u32 {
    const p = m *% @as(u64, n);
    return @intCast(mul128_u32(d, p));
}

pub inline fn fastmod_u64(n: u64, m: u128, d: u64) u64 {
    const p = m *% @as(u128, n);
    return mul128_u64(d, p);
}

pub inline fn mul128_u32(n: u32, m: u64) u64 {
    return @intCast((@as(u128, n) * @as(u128, m)) >> 64);
}

pub inline fn mul128_u64(n: u64, m: u128) u64 {
    const bottom_half = ((m & u64_max) * @as(u128, n)) >> 64;
    const top_half = (m >> 64) * @as(u128, n);
    return @intCast((bottom_half + top_half) >> 64);
}

pub inline fn magic_u32(d: u32) u64 {
    if (d == 1) return 1;
    return (u64_max) / @as(u64, d) + 1;
}

pub inline fn magic_u64(d: u64) u128 {
    if (d == 1) return 1;
    return (u128_max / @as(u128, d)) + 1;
}

pub inline fn is_divisible_u32(n: u32, m: u64) bool {
    return @as(u128, n) * m <= m - 1;
}

pub inline fn is_divisible_u64(n: u64, m: u128) bool {
    return @as(u128, n) * m <= m - 1;
}

fn MagicType(T: type) type {
    return switch (T) {
        u32 => MagicU32,
        u64 => MagicU64,
        else => @compileError("Unsupported divisor type " ++ @typeName(T)),
    };
}

fn ReverseMagicType(T: type) type {
    return switch (T) {
        MagicU32 => u32,
        MagicU64 => u64,
        else => @compileError("Unsupported magic type " ++ @typeName(T)),
    };
}

test "fastmod/fastdiv u32" {
    var r = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    var random = r.random();
    const n: u32 = 10000; // u32_max
    for (1..(@as(u64, n) + 1)) |du| {
        const d: u32 = @intCast(du);
        const v = random.uintAtMost(u32, u32_max - 1) + 1;
        const m = magicNumber(v);
        const div = fastdiv(m, d);
        std.testing.expectEqual(d / v, div) catch |err| {
            std.debug.print("{} {}\n", .{ d, v });
            return err;
        };
        const mod = fastmod(m, d, v);
        std.testing.expectEqual(d % v, mod) catch |err| {
            std.debug.print("{} {}\n", .{ d, v });
            return err;
        };
    }
}

test "fastmod/fastdiv u64" {
    var r = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    var random = r.random();

    const n = 10000;
    for (1..n) |du| {
        const d: u64 = @intCast(du);
        const v = random.uintAtMost(u64, u64_max - 1) + 1;
        const m = magicNumber(v);
        const div = fastdiv(m, d);
        std.testing.expectEqual(d / v, div) catch |err| {
            std.debug.print("{} {}\n", .{ d, v });
            return err;
        };
        const mod = fastmod(m, d, v);
        std.testing.expectEqual(d % v, mod) catch |err| {
            std.debug.print("{} {}\n", .{ d, v });
            return err;
        };
    }

    for (u64_max - n..u64_max) |du| {
        const d: u64 = @intCast(du);
        const v = random.uintAtMost(u64, u64_max - 1) + 1;
        const m = magicNumber(v);
        const div = fastdiv(m, d);
        std.testing.expectEqual(d / v, div) catch |err| {
            std.debug.print("{} {}\n", .{ d, v });
            return err;
        };
        const mod = fastmod(m, d, v);
        std.testing.expectEqual(d % v, mod) catch |err| {
            std.debug.print("{} {}\n", .{ d, v });
            return err;
        };
    }
}
