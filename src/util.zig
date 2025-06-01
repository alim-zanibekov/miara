const std = @import("std");
const builtin = @import("builtin");

pub fn toF64(it: anytype) f64 {
    return switch (@typeInfo(@TypeOf(it))) {
        .comptime_int, .int => @as(f64, @floatFromInt(it)),
        else => @compileError("as_f64 requires an unsigned integer, found " ++ @typeName(@TypeOf(it))),
    };
}

pub fn randomString(allocator: std.mem.Allocator, len: usize, rng: std.Random) ![]u8 {
    const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    const buffer = try allocator.alloc(u8, len);
    for (buffer) |*ch| {
        const idx = rng.uintLessThan(usize, charset.len);
        ch.* = charset[idx];
    }
    return buffer;
}

pub const RandomStrings = struct {
    strings: [][]u8,
    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        for (self.strings) |s| allocator.free(s);
        allocator.free(self.strings);
        self.* = undefined;
    }
};

pub fn randomStrings(allocator: std.mem.Allocator, n: usize, str_size: usize, rng: std.Random) !RandomStrings {
    var strings = try allocator.alloc([]u8, n);
    for (0..n) |i| {
        strings[i] = try randomString(allocator, str_size, rng);
    }
    return .{ .strings = strings };
}

pub fn isStringSlice(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .pointer => |ptr| switch (ptr.size) {
            .slice => isCharType(ptr.child),
            else => false,
        },
        .array => |arr| isCharType(arr.child),
        else => false,
    };
}

pub fn isCharType(comptime T: type) bool {
    return T == u8 or T == u16 or T == u32 or T == u21;
}
