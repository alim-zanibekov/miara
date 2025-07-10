// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Alim Zanibekov

const std = @import("std");
const builtin = @import("builtin");

/// Converts any integer to f64
pub fn toF64(it: anytype) f64 {
    return switch (@typeInfo(@TypeOf(it))) {
        .comptime_int, .int => @as(f64, @floatFromInt(it)),
        else => @compileError("as_f64 requires an unsigned integer, found " ++ @typeName(@TypeOf(it))),
    };
}

pub fn log2IntCeilOrZero(comptime T: type, x: T) std.math.Log2IntCeil(T) {
    if (x == 0) return 0;
    return std.math.log2_int_ceil(T, x);
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

pub fn PowerOf2Int(comptime T: type) type {
    const info = @typeInfo(T).int;
    const bits = std.math.ceilPowerOfTwo(u64, info.bits) catch unreachable;
    return std.meta.Int(info.signedness, bits);
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

/// Converts a byte count to a human-readable string, backed by comptime sized buffer to avoid dealing with allocators
/// Intended for testing purposes
pub const HumanBytes = struct {
    buffer: [1024]u8 = undefined,
    buffer_index: usize = 0,

    pub fn fmt(self: *@This(), bytes: u64) []const u8 {
        const units = [_][]const u8{ "b", "Kb", "Mb", "Gb", "Tb", "Pb" };

        const current_buffer = self.buffer[self.buffer_index..];

        if (bytes == 0) {
            return "0 b";
        }

        var size: f64 = @floatFromInt(bytes);
        var unit_index: usize = 0;

        while (size >= 1024.0 and unit_index < units.len - 1) {
            size /= 1024.0;
            unit_index += 1;
        }

        var result: []const u8 = undefined;

        if (unit_index == 0) {
            result = std.fmt.bufPrint(current_buffer, "{d} {s}", .{ bytes, units[unit_index] }) catch unreachable;
        } else {
            result = std.fmt.bufPrint(current_buffer, "{d:.2} {s}", .{ size, units[unit_index] }) catch unreachable;
        }

        self.buffer_index += result.len;

        return result;
    }
};

/// Converts time (duration) to a human-readable string, backed by comptime sized buffer to avoid dealing with allocators
/// Intended to be used for testing purposes
pub const HumanTime = struct {
    buffer: [1024]u8 = undefined,
    buffer_index: usize = 0,

    pub fn fmt(self: *@This(), duration: i128) []const u8 {
        const units = [_][]const u8{ "ns", "us", "ms", "sec", "min", "hr" };
        const thresholds = [_]f64{ 1000, 1000, 1000, 60, 60 };

        const current_buffer = self.buffer[self.buffer_index..];

        if (duration == 0) {
            return "0 ns";
        }

        var size: f64 = @floatFromInt(@abs(duration));
        var unit_index: usize = 0;

        while (unit_index < thresholds.len and size >= thresholds[unit_index]) {
            size /= thresholds[unit_index];
            unit_index += 1;
        }

        var result: []const u8 = undefined;

        if (unit_index == 0) {
            result = std.fmt.bufPrint(current_buffer, "{d} {s}", .{ duration, units[unit_index] }) catch unreachable;
        } else {
            const sign: f64 = if (duration < 0) -1.0 else 1.0;
            result = std.fmt.bufPrint(current_buffer, "{d:.2} {s}", .{ size * sign, units[unit_index] }) catch unreachable;
        }

        self.buffer_index += result.len;

        return result;
    }
};

/// Converts time (duration) to a human-readable string, backed by comptime sized buffer to avoid dealing with allocators
/// Intended to be used for testing purposes
pub const TimeMasurer = struct {
    time: i128 = 0,
    human_time: HumanTime = .{},

    pub fn start(self: *@This()) void {
        self.time = std.time.nanoTimestamp();
    }

    pub fn loop(self: *@This(), name: []const u8) void {
        std.debug.print("{s} took {s}\n", .{ name, self.human_time.fmt(std.time.nanoTimestamp() - self.time) });
        self.time = std.time.nanoTimestamp();
    }
};

/// Returns true if type T is composed entirely of static memory (no heap data)
pub fn isStaticType(comptime T: type) bool {
    switch (@typeInfo(T)) {
        .int, .float, .bool, .@"enum", .error_set, .void => return true,
        .array => |it| return isStaticType(it.child),
        .@"struct" => |it| {
            inline for (it.fields) |field| {
                if (!isStaticType(field.type)) return false;
            }
            return true;
        },
        .pointer => return false,
        .optional => |it| return isStaticType(it.child),
        .vector => |it| return isStaticType(it.child),
        .@"union" => |it| {
            if (it.tag_type == null) return false;
            inline for (it.fields) |field| {
                if (!isStaticType(field.type)) return false;
            }
            return true;
        },
        else => return false,
    }
}

/// Returns the runtime memory size (in bytes) of a value, accounting for nested slices/strings
pub fn calculateRuntimeSize(allocator: std.mem.Allocator, comptime T: type, value: T) !usize {
    const hint = comptime isStaticType(T);
    var pointer_set = std.AutoHashMap(*const anyopaque, void).init(allocator);
    defer pointer_set.deinit();
    switch (hint) {
        true => return @sizeOf(T),
        false => return try calculateRuntimeSizeImpl(T, value, &pointer_set),
    }
}

fn calculateRuntimeSizeImpl(comptime T: type, value: T, ps: *std.AutoHashMap(*const anyopaque, void)) !usize {
    var total_size: usize = 0;
    switch (@typeInfo(T)) {
        .@"fn" => return @sizeOf(T),
        .array => |info| {
            if (comptime isStaticType(info.child)) return @sizeOf(T);
            for (value) |item| {
                total_size += try calculateRuntimeSizeImpl(info.child, item, ps);
            }
            return total_size;
        },
        .pointer => |info| {
            switch (info.size) {
                .slice => {
                    total_size = @sizeOf(T);
                    if (ps.*.contains(@ptrCast(value.ptr))) return total_size;
                    try ps.*.put(@ptrCast(value.ptr), {});
                    if (comptime isStaticType(info.child)) return @sizeOf(T) + value.len * @sizeOf(info.child);
                    for (value) |item| {
                        total_size += try calculateRuntimeSizeImpl(info.child, item, ps);
                    }
                    return total_size;
                },
                .one => if (comptime @TypeOf(value) == T and @typeInfo(info.child) != .@"fn" and @typeInfo(info.child) != .@"opaque") {
                    if (ps.*.contains(@ptrCast(value))) return total_size;
                    try ps.*.put(@ptrCast(value), {});
                    return @sizeOf(T) + try calculateRuntimeSizeImpl(info.child, value.*, ps);
                } else {
                    return @sizeOf(T);
                },
                else => return @sizeOf(T),
            }
        },
        .@"struct" => |info| {
            total_size = @sizeOf(T);
            inline for (info.fields) |field| {
                if (!(comptime isStaticType(field.type))) {
                    total_size += try calculateRuntimeSizeImpl(field.type, @field(value, field.name), ps) -| @sizeOf(field.type);
                }
            }
            return total_size;
        },
        .optional => |info| {
            if (value) |unwrapped| {
                if (comptime isStaticType(info.child)) return @sizeOf(T);
                return @sizeOf(T) + try calculateRuntimeSizeImpl(info.child, unwrapped, ps) - @sizeOf(info.child);
            } else {
                return @sizeOf(T);
            }
        },
        .@"union" => |union_info| {
            if (union_info.tag_type) |_| {
                if (comptime isStaticType(T)) return @sizeOf(T);
                total_size = @sizeOf(T);
                switch (value) {
                    inline else => |union_value| {
                        const union_field_type = @TypeOf(union_value);
                        total_size += if (comptime isStaticType(T))
                            0
                        else
                            try calculateRuntimeSizeImpl(union_field_type, union_value, ps) - @sizeOf(union_field_type);
                    },
                }
                return total_size;
            }
            return @sizeOf(T);
        },
        else => return @sizeOf(T),
    }
}

const testing = std.testing;

test "static array" {
    const arr = [_]i32{ 1, 2, 3 };
    const size = try calculateRuntimeSize(testing.allocator, [3]i32, arr);
    try testing.expectEqual(@sizeOf([3]i32), size);
}

test "slice of ints" {
    const arr = [_]i32{ 1, 2, 3 };
    const slice: []const i32 = &arr;
    const size = try calculateRuntimeSize(testing.allocator, []const i32, slice);
    try testing.expectEqual(@sizeOf([]const i32) + 3 * @sizeOf(i32), size);
}

test "slice of strings" {
    const strings = [_][]const u8{ "hi", "bye" };
    const slice: []const []const u8 = &strings;
    const size = try calculateRuntimeSize(testing.allocator, []const []const u8, slice);
    const expected = @sizeOf([]const []const u8) +
        (@sizeOf([]const u8) + 2) +
        (@sizeOf([]const u8) + 3);
    try testing.expectEqual(expected, size);
}

test "struct with slice" {
    const data = [_]i32{ 1, 2 };
    const s = struct { name: []const u8, nums: []const i32 }{
        .name = "test",
        .nums = &data,
    };
    const size = try calculateRuntimeSize(testing.allocator, @TypeOf(s), s);
    try testing.expect(size > 0);
}

test "empty slice" {
    const slice: []const i32 = &[_]i32{};
    const size = try calculateRuntimeSize(testing.allocator, []const i32, slice);
    try testing.expectEqual(@sizeOf([]const i32), size);
}

test "struct with nested slice" {
    const Inner = struct {
        values: []const i32,
    };

    const Outer = struct {
        inner: Inner,
        value: u32,
        tag: []const u8,
    };

    const arr = [_]i32{ 1, 2 };
    const outer = Outer{
        .inner = Inner{ .values = &arr },
        .tag = "test",
        .value = 666,
    };

    const size = try calculateRuntimeSize(testing.allocator, Outer, outer);
    const expected = @sizeOf(Outer) + (2 * @sizeOf(i32)) + 4;
    try testing.expectEqual(expected, size);
}

test "struct with same pointer twice" {
    const Struct = struct {
        values1: []const i32,
        values2: []const i32,
        v1: *const std.AutoArrayHashMapUnmanaged(u32, u32),
        v2: *const std.AutoArrayHashMapUnmanaged(u32, u32),
    };

    const arr = &[_]i32{ 1, 2 };
    var map = std.AutoArrayHashMapUnmanaged(u32, u32){};
    const st = Struct{
        .values1 = arr,
        .values2 = arr,
        .v1 = &map,
        .v2 = &map,
    };

    const size = try calculateRuntimeSize(testing.allocator, Struct, st);
    const expected = @sizeOf(Struct) + 2 * @sizeOf(i32) + @sizeOf(@TypeOf(map));
    try testing.expectEqual(expected, size);
}
