// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Alim Zanibekov

const std = @import("std");
const iface = @import("interface.zig");

/// Defines a minimal forward `Iterator` interface for a given element type `T`
/// Intended only as a comptime reference for `interface.zig` -> `checkImplements`
pub fn Iterator(T: type) type {
    return struct {
        pub fn size(_: *const @This()) usize {
            unreachable;
        }

        pub fn next(_: *@This()) ?T {
            unreachable;
        }
    };
}

/// Defines a reusable forward `Iterator` interface for a given element type `T`
/// Intended only as a comptime reference for `interface.zig` -> `checkImplements`
pub fn ReusableIterator(T: type) type {
    return struct {
        pub fn size(_: *const @This()) usize {
            unreachable;
        }

        pub fn next(_: *@This()) ?T {
            unreachable;
        }

        pub fn reset(_: *@This()) void {
            unreachable;
        }
    };
}

/// Defines a `RandomAccessIterator` interface supporting indexing, movement, and reverse iteration
/// Intended only as a comptime reference for `interface.zig` -> `checkImplements`
pub fn RandomAccessIterator(T: type) type {
    return struct {
        pub fn size(_: *const @This()) usize {
            unreachable;
        }

        pub fn next(_: *@This()) ?T {
            unreachable;
        }

        pub fn prev(_: *@This()) ?T {
            unreachable;
        }

        pub fn move(_: *@This(), _: isize) void {
            unreachable;
        }

        pub fn get(_: *@This(), _: usize) T {
            unreachable;
        }
    };
}

/// `RandomAccessIterator` over a slice of `T`
pub fn SliceIterator(T: type) type {
    return struct {
        const Self = @This();
        const Child = T;

        array: []const T,
        i: isize = 0,

        pub fn init(array: []const T) Self {
            return .{ .array = array };
        }

        pub fn size(self: *const Self) usize {
            return self.array.len;
        }

        pub fn next(self: *Self) ?T {
            if (self.i >= self.array.len) return null;
            const res = self.array[@intCast(self.i)];
            self.i += 1;
            return res;
        }

        pub fn prev(self: *Self) ?T {
            if (self.i < 0) return null;
            const res = self.array[@intCast(self.i)];
            self.i -= 1;
            return res;
        }

        pub fn move(self: *Self, i: isize) void {
            self.i = self.i + i;
        }

        pub fn get(self: *Self, i: usize) T {
            return self.array[i];
        }

        pub fn reset(self: *Self) void {
            self.i = 0;
        }
    };
}

test "SliceIteratorChech" {
    try std.testing.expect(comptime iface.checkImplements(RandomAccessIterator(u64), SliceIterator(u64)));
}
