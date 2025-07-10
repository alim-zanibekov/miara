const std = @import("std");
const builtin = @import("builtin");
const native_endian = builtin.cpu.arch.endian();

/// Serialize value using native endianness. Convenience wrapper for serializeEndian
/// with platform-specific byte ordering. Returns total bytes written to stream
pub fn serialize(allocator: std.mem.Allocator, writer: anytype, value: anytype) !usize {
    return serializeEndian(allocator, writer, value, native_endian);
}

/// Deserialize value using native endianness. Convenience wrapper for deserializeEndian
/// with platform-specific byte ordering. Returns reconstructed object and DeDeinit struct
/// to free the allocated memory
pub fn deserialize(allocator: std.mem.Allocator, reader: anytype, comptime T: type) !struct { T, DeDeinit } {
    return deserializeEndian(allocator, reader, T, native_endian);
}

/// Serialize value with specified endianness. Writes binary header field count,
/// type count, and type info size. Processes type information and object
/// data while managing pointer offsets for self-referencing structures. Supports
/// circular references and complex object graphs through pointer tracking system
pub fn serializeEndian(allocator: std.mem.Allocator, writer: anytype, value: anytype, endian: std.builtin.Endian) !usize {
    return serializeEndianImpl(allocator, writer, value, endian);
}

/// Deserialize value with specified endianness. Validates binary header format,
/// reads type and field information using temporary arena allocator, then
/// reconstructs the object with proper memory management. Handles self-referencing
/// objects and circular references through pointer tracking system that maintains
/// object identity during reconstruction. Returns reconstructed object and DeDeinit struct
/// to free the allocated memory
pub fn deserializeEndian(allocator: std.mem.Allocator, reader: anytype, comptime T: type, endian: std.builtin.Endian) !struct { T, DeDeinit } {
    return deserializeEndianImpl(allocator, reader, T, endian);
}

/// Serialize value with specified endianness at compile time
pub fn serializeEndianComptime(writer: anytype, value: anytype, endian: std.builtin.Endian) !usize {
    comptime {
        const ca = ComptimeAllocator{};
        return serializeEndianImpl(ca, writer, value, endian);
    }
}

/// Deserialize value with specified endianness at compile time
/// WARNING: The resulting object is not transferrable to runtime if it contains pointers,
/// since it will have references to comptime mutable memory
pub fn deserializeEndianComptime(reader: anytype, comptime T: type, endian: std.builtin.Endian) !T {
    comptime {
        return try deserializeEndianImplComptime(reader, T, endian);
    }
}

/// Pretty print file type. Wrapper for  prettyPrintFileTypeEndian with platform-native byte ordering
pub fn prettyPrintFileType(
    allocator: std.mem.Allocator,
    input: anytype,
    output: anytype,
    indent: []const u8,
    linebreak: []const u8,
) !void {
    return prettyPrintFileTypeEndian(allocator, input, native_endian, output, indent, linebreak);
}

/// Pretty print file type with specified endianness. Outputs human-readable formatted
/// type structure with proper indentation and line breaks. Can handle self-referencing structures
pub fn prettyPrintFileTypeEndian(
    allocator: std.mem.Allocator,
    input: anytype,
    endian: std.builtin.Endian,
    output: anytype,
    indent: []const u8,
    linebreak: []const u8,
) !void {
    const header, _ = try readInt(input, u16, endian);
    if (header != 0x0001) return error.InvalidFormat;
    const fields_len, _ = try readInt(input, u32, endian);
    const types_len, _ = try readInt(input, u32, endian);
    _, _ = try readInt(input, u32, endian);

    var arena = if (!@inComptime()) std.heap.ArenaAllocator.init(allocator);
    defer if (!@inComptime()) arena.deinit();
    const tmp_allocator = if (!@inComptime()) arena.allocator() else allocator;

    const fields = try tmp_allocator.alloc(Type.Field, fields_len);
    var fields_idx: usize = 0;
    var offsets = HashMapRuntime(u32, *const Type).init(allocator);
    defer offsets.deinit();
    try offsets.map.ensureUnusedCapacity(types_len);
    const t, _ = try readTypeInfo(input, 0, tmp_allocator, fields, &fields_idx, &offsets, endian);
    try prettyPrintType(allocator, t, fields, output, 0, indent, linebreak);
    try output.writeAll(linebreak);
}

fn serializeEndianImpl(allocator: anytype, writer: anytype, value: anytype, endian: std.builtin.Endian) !usize {
    const T = @TypeOf(value);
    var written: usize = 0;

    const t, const fields, const n_types, const type_info_size = comptime toTypeInfo(T);

    written += try writeInt(writer, u16, 0x0001, endian);
    written += try writeInt(writer, u32, fields.len, endian);
    written += try writeInt(writer, u32, n_types, endian);
    written += try writeInt(writer, u32, type_info_size, endian);

    var offsets: [n_types]TypeOffset, var offsets_idx: usize = .{ undefined, 0 };
    written += try writeTypeInfo(writer, 0, t, &fields, &offsets, &offsets_idx, endian);
    if (@inComptime()) {
        var pointers_l = HashMapComptime(*anyopaque, PointerOffset){};
        written += try writeObject(writer, 0, &pointers_l, value, t, &fields, endian);
        return written;
    } else {
        var pointers_l = HashMapRuntime(*anyopaque, PointerOffset).init(allocator);
        defer pointers_l.deinit();
        written += try writeObject(writer, 0, &pointers_l, value, t, &fields, endian);
        return written;
    }
}

fn deserializeEndianImpl(allocator: anytype, reader: anytype, comptime T: type, endian: std.builtin.Endian) !struct { T, DeDeinit } {
    var read: usize = 0;

    const header, const r1 = try readInt(reader, u16, endian);
    if (header != 0x0001) return error.InvalidFormat;

    const fields_len, const r2 = try readInt(reader, u32, endian);
    const types_len, const r3 = try readInt(reader, u32, endian);
    _, const r4 = try readInt(reader, u32, endian);
    read += r1 + r2 + r3 + r4;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const tmp_allocator = arena.allocator();

    const fields = try tmp_allocator.alloc(Type.Field, fields_len);
    var fields_idx: usize = 0;
    var offsets = HashMapRuntime(u32, *const Type).init(allocator);
    defer offsets.deinit();
    try offsets.map.ensureUnusedCapacity(types_len);
    const t, var r = try readTypeInfo(reader, 0, tmp_allocator, fields, &fields_idx, &offsets, endian);
    read += r;
    var pointers_l = HashMapRuntime(u64, PointerOffset).init(allocator);
    errdefer DeDeinit.deinitHeadless(allocator, &pointers_l);
    const value, r = try readObject(reader, 0, allocator, &pointers_l, T, t, fields, endian);
    read += r;
    return .{ value, DeDeinit.init(pointers_l) };
}

fn deserializeEndianImplComptime(reader: anytype, comptime T: type, endian: std.builtin.Endian) !T {
    const header, _ = try readInt(reader, u16, endian);
    if (header != 0x0001) return error.InvalidFormat;

    const fields_len, _ = try readInt(reader, u32, endian);
    _, _ = try readInt(reader, u32, endian);
    _, _ = try readInt(reader, u32, endian);

    const allocator = ComptimeAllocator{};
    const fields = try allocator.alloc(Type.Field, fields_len);
    var fields_idx: usize = 0;
    var offsets = HashMapComptime(u32, *const Type){};
    const t, _ = try readTypeInfo(reader, 0, allocator, fields, &fields_idx, &offsets, endian);

    var pointers_l = HashMapComptime(u64, PointerOffset){};
    const value, _ = try readObject(reader, 0, allocator, &pointers_l, T, t, fields, endian);

    return value;
}

const ListDoubly = std.DoublyLinkedList(void);
const ListSingly = std.SinglyLinkedList(void);

const LIST_D_NODE_NULL = ListDoubly.Node{ .data = {} };
const LIST_S_NODE_NULL = ListSingly.Node{ .data = {} };

const DeDeinit = struct {
    pointers: HashMapRuntime(u64, PointerOffset),

    fn init(pointers: HashMapRuntime(u64, PointerOffset)) @This() {
        return .{ .pointers = pointers };
    }

    fn deinitHeadless(allocator: std.mem.Allocator, pointers: *HashMapRuntime(u64, PointerOffset)) void {
        var it = pointers.map.valueIterator();
        while (it.next()) |entry| {
            if (entry.len > 0) {
                allocator.rawFree(@as([*]u8, @ptrCast(@alignCast(entry.ptr)))[0..entry.len], entry.alignment, @returnAddress());
            }
        }
        pointers.deinit();
    }

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        deinitHeadless(allocator, &self.pointers);
        self.* = undefined;
    }
};

const TypeEntry = struct {
    type: *const Type,
    list_node: ListDoubly.Node = LIST_D_NODE_NULL,
};

const InventoryFwdEntry = struct {
    type: *const Type,
    zig_type: type,
    list_node: ListSingly.Node = LIST_S_NODE_NULL,
};

const CmpPair = struct {
    type_a: *const Type,
    type_b: *const Type,
    list_node: ListSingly.Node = LIST_S_NODE_NULL,
};

const PointerOffset = struct {
    ptr: *anyopaque,
    len: usize = 0,
    alignment: std.mem.Alignment = .@"1",
    offset: u64,
};

const TypeOffset = struct {
    offset: u32,
    type: *const Type,
};

pub const TypeId = enum { void, bool, int, float, array, vector, pointer, optional, @"enum", @"union", @"struct" };

pub const Type = union(TypeId) {
    void: void,
    bool: void,
    int: Int,
    float: Float,
    array: Array,
    vector: Vector,
    pointer: Pointer,
    optional: Optional,
    @"enum": Enum,
    @"union": Union,
    @"struct": Struct,

    pub const Int = struct { signedness: std.builtin.Signedness, bits: u16 };

    pub const Float = struct { bits: u16 };

    pub const Array = struct { len: u64, child: *const Type };

    pub const Pointer = struct { child: *const Type };

    pub const Vector = struct { child: *const Type };

    pub const Optional = struct { child: *const Type };

    pub const Field = union {
        @"enum": FieldEnum,
        @"union": FieldStruct,
        @"struct": FieldUnion,

        pub const FieldEnum = struct {
            name: []const u8,
            // TODO: allow custom value type?
            value: u32,
        };

        pub const FieldStruct = struct {
            name: []const u8,
            type: *const Type,
        };

        pub const FieldUnion = struct {
            name: []const u8,
            type: *const Type,
        };
    };

    pub const Enum = struct {
        fields_idx: u64,
        fields_len: usize,
    };

    pub const Struct = struct {
        fields_idx: u64,
        fields_len: usize,
    };

    pub const Union = struct {
        fields_idx: u64,
        fields_len: usize,
    };
};

pub fn isTypesEqual(
    a: *const Type,
    fields_a: []const Type.Field,
    b: *const Type,
    fields_b: []const Type.Field,
) bool {
    var fwd_inv = ListSingly{};
    return isTypesEqualImpl(a, fields_a, b, fields_b, &fwd_inv);
}

fn isTypesEqualImpl(
    a: *const Type,
    fields_a: []const Type.Field,
    b: *const Type,
    fields_b: []const Type.Field,
    fwd_inv: *ListSingly,
) bool {
    if (a == b) return true;
    if (std.meta.activeTag(a.*) != std.meta.activeTag(b.*)) return false;

    {
        var node = fwd_inv.first;
        while (node) |n| {
            const entry: *CmpPair = @fieldParentPtr("list_node", n);
            if (entry.type_a == a and entry.type_b == b) return true;
            node = n.next;
        }
    }

    var entry = CmpPair{ .type_a = a, .type_b = b };
    fwd_inv.prepend(&entry.list_node);
    defer _ = fwd_inv.popFirst();

    switch (a.*) {
        .void => return true,
        .bool => return true,
        .int => |it| return it.signedness == b.int.signedness and it.bits == b.int.bits,
        .float => |it| return it.bits == b.float.bits,
        .array => |it| return it.len == b.array.len and isTypesEqualImpl(it.child, fields_a, b.array.child, fields_b, fwd_inv),
        .vector => |it| return isTypesEqualImpl(it.child, fields_a, b.vector.child, fields_b, fwd_inv),
        .optional => |it| return isTypesEqualImpl(it.child, fields_a, b.optional.child, fields_b, fwd_inv),
        .pointer => |it| return isTypesEqualImpl(it.child, fields_a, b.pointer.child, fields_b, fwd_inv),
        .@"enum" => |it_a| {
            const it_b = b.@"enum";
            if (it_a.fields_len != it_b.fields_len) return false;
            for (
                it_a.fields_idx..(it_a.fields_idx + it_a.fields_len),
                it_b.fields_idx..(it_b.fields_idx + it_b.fields_len),
            ) |i_a, i_b| {
                const fa = fields_a[i_a].@"enum";
                const fb = fields_b[i_b].@"enum";
                if (!std.mem.eql(u8, fa.name, fb.name) or fa.value != fb.value) return false;
            }
            return true;
        },
        .@"union" => |it_a| {
            const it_b = b.@"union";
            if (it_a.fields_len != it_b.fields_len) return false;
            for (
                it_a.fields_idx..(it_a.fields_idx + it_a.fields_len),
                it_b.fields_idx..(it_b.fields_idx + it_b.fields_len),
            ) |i_a, i_b| {
                const fa = fields_a[i_a].@"union";
                const fb = fields_b[i_b].@"union";
                if (!std.mem.eql(u8, fa.name, fb.name) or !isTypesEqualImpl(fa.type, fields_a, fb.type, fields_b, fwd_inv)) return false;
            }
            return true;
        },
        .@"struct" => |it_a| {
            const it_b = b.@"struct";
            if (it_a.fields_len != it_b.fields_len) return false;
            for (
                it_a.fields_idx..(it_a.fields_idx + it_a.fields_len),
                it_b.fields_idx..(it_b.fields_idx + it_b.fields_len),
            ) |i_a, i_b| {
                const fa = fields_a[i_a].@"struct";
                const fb = fields_b[i_b].@"struct";
                if (!std.mem.eql(u8, fa.name, fb.name) or !isTypesEqualImpl(fa.type, fields_a, fb.type, fields_b, fwd_inv)) return false;
            }
            return true;
        },
    }
}

fn toTypeInfo(comptime T: type) lb: {
    var tif = ListSingly{};
    const size = toTypeInfoFieldsSize(T, &tif);
    break :lb struct { *const Type, [size]Type.Field, usize, usize };
} {
    comptime {
        var type_inventory_backward = ListDoubly{};
        var type_inventory_forward = ListSingly{};
        const size = toTypeInfoFieldsSize(T, &type_inventory_forward);
        var fields = [_]Type.Field{undefined} ** size;
        type_inventory_forward = ListSingly{};
        var fs_i: usize = 0;
        const tp = toTypeInfoImpl(T, &fields, &fs_i, &type_inventory_forward, &type_inventory_backward);
        var offsets: [type_inventory_backward.len]TypeOffset, var offsets_idx: usize = .{ undefined, 0 };
        const cap = serializedTypeInfoSize(0, tp, &fields, &offsets, &offsets_idx);
        return .{ tp, fields, type_inventory_backward.len, cap };
    }
}

fn toTypeInfoFieldsSize(
    comptime T: type,
    comptime fwd_inv: *ListSingly,
) usize {
    comptime {
        {
            var node = fwd_inv.first;
            while (node) |n| {
                const entry: *InventoryFwdEntry = @fieldParentPtr("list_node", n);
                if (T == entry.zig_type) {
                    return 0;
                }
                node = n.next;
            }
        }

        const result = switch (@typeInfo(T)) {
            .optional => |it| toTypeInfoFieldsSize(it.child, fwd_inv),
            .array => |it| toTypeInfoFieldsSize(it.child, fwd_inv),
            .vector => |it| toTypeInfoFieldsSize(it.child, fwd_inv),
            .pointer => |it| switch (it.size) {
                .one, .slice => toTypeInfoFieldsSize(it.child, fwd_inv),
                .many, .c => @compileError("Unsupported slice type"),
            },
            .@"enum" => |it| lb: {
                break :lb it.fields.len;
            },
            .@"union" => |it| lb: {
                if (it.tag_type == null) @compileError("Unsupported type `untagged union`");
                var entry = InventoryFwdEntry{ .type = &Type{ .void = {} }, .zig_type = T };
                fwd_inv.prepend(&entry.list_node);
                var size: usize = 0;
                for (it.fields) |f| size += 1 + toTypeInfoFieldsSize(f.type, fwd_inv);
                break :lb size;
            },
            .@"struct" => |it| lb: {
                var entry = InventoryFwdEntry{ .type = &Type{ .void = {} }, .zig_type = T };
                fwd_inv.prepend(&entry.list_node);
                var size: usize = 0;
                for (it.fields) |f| size += 1 + toTypeInfoFieldsSize(f.type, fwd_inv);
                break :lb size;
            },
            else => 0,
        };

        return result;
    }
}

fn toTypeInfoImpl(
    comptime T: type,
    comptime fs: []Type.Field,
    comptime fs_i: *usize,
    comptime fwd_inv: *ListSingly,
    comptime bcw_inv: *ListDoubly,
) *const Type {
    comptime {
        {
            // Dodge recursive structs
            var node = fwd_inv.first;
            while (node) |n| {
                const entry: *InventoryFwdEntry = @fieldParentPtr("list_node", n);
                if (T == entry.zig_type) return entry.type;
                node = n.next;
            }
        }

        const result = switch (@typeInfo(T)) {
            .void => &Type{ .void = {} },
            .bool => &Type{ .bool = {} },
            .int => |it| &Type{ .int = .{ .bits = it.bits, .signedness = it.signedness } },
            .float => |it| &Type{ .float = .{ .bits = it.bits } },
            .array => |it| &Type{ .array = .{ .len = it.len, .child = toTypeInfoImpl(it.child, fs, fs_i, fwd_inv, bcw_inv) } },
            .vector => |it| &Type{ .array = .{ .len = it.len, .child = toTypeInfoImpl(it.child, fs, fs_i, fwd_inv, bcw_inv) } },
            .pointer => |it| switch (it.size) {
                .one => &Type{ .pointer = .{ .child = toTypeInfoImpl(it.child, fs, fs_i, fwd_inv, bcw_inv) } },
                .slice => &Type{ .vector = .{ .child = toTypeInfoImpl(it.child, fs, fs_i, fwd_inv, bcw_inv) } },
                .many, .c => @compileError("Unsupported slice type"),
            },
            .optional => |it| &Type{ .optional = .{ .child = toTypeInfoImpl(it.child, fs, fs_i, fwd_inv, bcw_inv) } },
            .@"enum" => |it| lb: {
                const fields_idx = fs_i.*;
                for (it.fields) |f| {
                    fs[fs_i.*] = .{ .@"enum" = .{ .name = f.name, .value = f.value } };
                    fs_i.* += 1;
                }
                break :lb &Type{ .@"enum" = .{ .fields_idx = fields_idx, .fields_len = it.fields.len } };
            },
            .@"union" => |it| lb: {
                if (it.tag_type == null) @compileError("Unsupported type `untagged union`");
                const fields_idx = fs_i.*;
                const t = &Type{ .@"union" = .{ .fields_idx = fields_idx, .fields_len = it.fields.len } };
                var entry = InventoryFwdEntry{ .type = t, .zig_type = T };
                fwd_inv.prepend(&entry.list_node);
                fs_i.* += it.fields.len;
                for (it.fields, 0..) |f, i| {
                    fs[fields_idx + i] = .{
                        .@"union" = .{ .name = f.name, .type = toTypeInfoImpl(f.type, fs, fs_i, fwd_inv, bcw_inv) },
                    };
                }
                break :lb t;
            },
            .@"struct" => |it| lb: {
                const fields_idx = fs_i.*;
                const t = &Type{ .@"struct" = .{ .fields_idx = fields_idx, .fields_len = it.fields.len } };
                var entry = InventoryFwdEntry{ .type = t, .zig_type = T };
                fwd_inv.prepend(&entry.list_node);
                fs_i.* += it.fields.len;
                for (it.fields, 0..) |f, i| {
                    fs[fields_idx + i] = .{
                        .@"struct" = .{ .name = f.name, .type = toTypeInfoImpl(f.type, fs, fs_i, fwd_inv, bcw_inv) },
                    };
                }

                break :lb t;
            },
            else => {
                @compileError("Unsupported type " ++ @typeName(T));
            },
        };

        {
            var node = bcw_inv.first;
            while (node) |n| {
                const entry: *TypeEntry = @fieldParentPtr("list_node", n);
                if (isTypesEqual(entry.type, fs, result, fs)) return entry.type;
                node = n.next;
            }
        }
        {
            var entry = TypeEntry{ .type = result };
            bcw_inv.append(&entry.list_node);
            return result;
        }
    }
}

fn serializedTypeInfoSize(
    offset_out: usize,
    t: *const Type,
    fs: []const Type.Field,
    of: []TypeOffset,
    of_i: *usize,
) usize {
    var offset = offset_out;
    {
        for (of[0..of_i.*]) |it| {
            if (isTypesEqual(it.type, fs, t, fs)) return offset + 5;
        }
    }

    of[of_i.*] = TypeOffset{ .offset = @intCast(offset), .type = t };
    of_i.* += 1;
    offset += 1;
    switch (t.*) {
        .void, .bool => {},
        .int => offset += 3,
        .float => offset += 2,
        .array => |it| {
            offset += 8;
            offset = serializedTypeInfoSize(offset, it.child, fs, of, of_i);
        },
        .vector => |it| offset = serializedTypeInfoSize(offset, it.child, fs, of, of_i),
        .optional => |it| offset = serializedTypeInfoSize(offset, it.child, fs, of, of_i),
        .pointer => |it| offset = serializedTypeInfoSize(offset, it.child, fs, of, of_i),
        .@"enum" => |it| {
            offset += 4;
            for (it.fields_idx..(it.fields_idx + it.fields_len)) |fi| {
                const f = fs[fi].@"enum";
                offset += 4 + 2 + f.name.len;
            }
        },
        .@"union" => |it| {
            offset += 4;
            for (it.fields_idx..(it.fields_idx + it.fields_len)) |fi| {
                const f = fs[fi].@"union";
                offset += 2 + f.name.len;
                offset = serializedTypeInfoSize(offset, f.type, fs, of, of_i);
            }
        },
        .@"struct" => |it| {
            offset += 4;
            for (it.fields_idx..(it.fields_idx + it.fields_len)) |fi| {
                const f = fs[fi].@"struct";
                offset += 2 + f.name.len;
                offset = serializedTypeInfoSize(offset, f.type, fs, of, of_i);
            }
        },
    }

    return offset;
}

fn writeTypeInfo(
    out: anytype,
    offset_out: usize,
    t: *const Type,
    fs: []const Type.Field,
    of: []TypeOffset,
    of_i: *usize,
    endian: std.builtin.Endian,
) (@TypeOf(out).Error || error{EndOfStream})!usize {
    const tag: u8 = @intFromEnum(std.meta.activeTag(t.*));
    var offset = offset_out;
    {
        for (of[0..of_i.*]) |it| {
            if (isTypesEqual(it.type, fs, t, fs)) {
                try out.writeByte(0xFF);
                offset += 1;
                offset += try writeInt(out, @TypeOf(it.offset), it.offset, endian);
                return offset;
            }
        }
    }

    try out.writeByte(tag);
    offset += 1;

    of[of_i.*] = TypeOffset{ .offset = @intCast(offset), .type = t };
    of_i.* += 1;

    switch (t.*) {
        .void => {},
        .bool => {},
        .int => |it| {
            try out.writeByte(switch (it.signedness) {
                .unsigned => 0,
                .signed => 1,
            });
            offset += 1;
            offset += try writeInt(out, @TypeOf(it.bits), it.bits, endian);
        },
        .float => |it| offset += try writeInt(out, @TypeOf(it.bits), it.bits, endian),
        .array => |it| {
            offset += try writeInt(out, @TypeOf(it.len), it.len, endian);
            offset = try writeTypeInfo(out, offset, it.child, fs, of, of_i, endian);
        },
        .vector => |it| offset = try writeTypeInfo(out, offset, it.child, fs, of, of_i, endian),
        .optional => |it| offset = try writeTypeInfo(out, offset, it.child, fs, of, of_i, endian),
        .pointer => |it| offset = try writeTypeInfo(out, offset, it.child, fs, of, of_i, endian),
        .@"enum" => |it| {
            offset += try writeInt(out, u32, @intCast(it.fields_len), endian); // len
            for (it.fields_idx..(it.fields_idx + it.fields_len)) |fi| {
                const f = fs[fi].@"enum";
                const f_len: u16 = @intCast(f.name.len);
                offset += try writeInt(out, @TypeOf(f.value), f.value, endian); // value
                offset += try writeInt(out, @TypeOf(f_len), f_len, endian); // name len
                try out.writeAll(f.name); // name
                offset += f.name.len;
            }
        },
        .@"union" => |it| {
            offset += try writeInt(out, u32, @intCast(it.fields_len), endian); // len
            for (it.fields_idx..(it.fields_idx + it.fields_len)) |fi| {
                const f = fs[fi].@"union";
                const f_len: u16 = @intCast(f.name.len);
                offset += try writeInt(out, @TypeOf(f_len), f_len, endian); // name len
                try out.writeAll(f.name); // name
                offset += f.name.len;
                offset = try writeTypeInfo(out, offset, f.type, fs, of, of_i, endian);
            }
        },
        .@"struct" => |it| {
            offset += try writeInt(out, u32, @intCast(it.fields_len), endian); // len
            for (it.fields_idx..(it.fields_idx + it.fields_len)) |fi| {
                const f = fs[fi].@"struct";
                const f_len: u16 = @intCast(f.name.len);
                offset += try writeInt(out, @TypeOf(f_len), f_len, endian); // name len
                try out.writeAll(f.name); // name
                offset += f.name.len;
                offset = try writeTypeInfo(out, offset, f.type, fs, of, of_i, endian);
            }
        },
    }

    return offset;
}

fn readTypeInfo(
    in: anytype,
    offset_in: usize,
    allocator: anytype,
    fs: []Type.Field,
    fs_i: *usize,
    offsets: anytype,
    endian: std.builtin.Endian,
) (@TypeOf(in).Error || error{ InvalidTypeInfo, EndOfStream } || @TypeOf(allocator).Error || UnPtr(@TypeOf(offsets)).Error)!struct { *const Type, usize } {
    var offset = offset_in;
    const tag_int = try in.readByte();
    offset += 1;
    if (tag_int == 0xFF) {
        const offset_type, const read = try readInt(in, u32, endian);
        offset += read;
        const entry = offsets.get(offset_type) orelse return error.InvalidTypeInfo;
        return .{ entry, offset };
    }

    const tag = std.meta.intToEnum(TypeId, tag_int) catch return error.InvalidTypeInfo;
    const t = try allocator.create(Type);

    try offsets.put(@intCast(offset), t);

    switch (tag) {
        .void => t.* = Type{ .void = {} },
        .bool => t.* = Type{ .bool = {} },
        .int => {
            const sign: std.builtin.Signedness = switch (try in.readByte()) {
                0 => .unsigned,
                1 => .signed,
                else => return error.InvalidTypeInfo,
            };
            offset += 1;
            const bits, const read = try readInt(in, u16, endian);
            offset += read;
            t.* = Type{ .int = .{ .bits = bits, .signedness = sign } };
        },
        .float => {
            const bits, const read = try readInt(in, u16, endian);
            offset += read;
            t.* = Type{ .float = .{ .bits = bits } };
        },
        .array => {
            const len, const read = try readInt(in, u64, endian);
            offset += read;
            const t_array, offset = try readTypeInfo(in, offset, allocator, fs, fs_i, offsets, endian);
            t.* = Type{ .array = .{ .len = len, .child = t_array } };
        },
        .vector => {
            const t_child, offset = try readTypeInfo(in, offset, allocator, fs, fs_i, offsets, endian);
            t.* = Type{ .vector = .{ .child = t_child } };
        },
        .optional => {
            const t_child, offset = try readTypeInfo(in, offset, allocator, fs, fs_i, offsets, endian);
            t.* = Type{ .optional = .{ .child = t_child } };
        },
        .pointer => {
            const t_child, offset = try readTypeInfo(in, offset, allocator, fs, fs_i, offsets, endian);
            t.* = Type{ .pointer = .{ .child = t_child } };
        },
        .@"enum" => {
            const fields_len, var read = try readInt(in, u32, endian);
            offset += read;
            const fields_idx = fs_i.*;
            fs_i.* += fields_len;
            for (0..fields_len) |i| {
                const f_val, read = try readInt(in, u32, endian);
                offset += read;
                const f_len, read = try readInt(in, u16, endian);
                offset += read;
                const buf = try allocator.alloc(u8, f_len);
                offset += try in.read(buf);
                fs[fields_idx + i] = .{ .@"enum" = .{ .name = buf, .value = f_val } };
            }
            t.* = Type{ .@"enum" = .{ .fields_idx = fields_idx, .fields_len = fields_len } };
        },
        .@"union" => {
            const fields_len, var read = try readInt(in, u32, endian);
            offset += read;
            const fields_idx = fs_i.*;
            fs_i.* += fields_len;
            for (0..fields_len) |i| {
                const f_len, read = try readInt(in, u16, endian);
                offset += read;
                const buf = try allocator.alloc(u8, f_len);
                offset += try in.read(buf);
                const t_union, offset = try readTypeInfo(in, offset, allocator, fs, fs_i, offsets, endian);
                fs[fields_idx + i] = .{ .@"union" = .{ .name = buf, .type = t_union } };
            }
            t.* = Type{ .@"union" = .{ .fields_idx = fields_idx, .fields_len = fields_len } };
        },
        .@"struct" => {
            const fields_len, var read = try readInt(in, u32, endian);
            offset += read;
            const fields_idx = fs_i.*;
            fs_i.* += fields_len;
            for (0..fields_len) |i| {
                const f_len, read = try readInt(in, u16, endian);
                offset += read;
                const buf = try allocator.alloc(u8, f_len);
                offset += try in.read(buf);
                const t_struct, offset = try readTypeInfo(in, offset, allocator, fs, fs_i, offsets, endian);
                fs[fields_idx + i] = .{ .@"struct" = .{ .name = buf, .type = t_struct } };
            }
            t.* = Type{ .@"struct" = .{ .fields_idx = fields_idx, .fields_len = fields_len } };
        },
    }

    return .{ t, offset };
}

fn writeObject(
    out: anytype,
    offset_out: usize,
    pointers: anytype,
    val: anytype,
    t: *const Type,
    fs: []const Type.Field,
    endian: std.builtin.Endian,
) (@TypeOf(out).Error || error{InvalidTypeInfo} || UnPtr(@TypeOf(pointers)).Error)!usize {
    var offset: usize = offset_out;
    const T = @TypeOf(val);

    // @compileLog("writeObject", T, offset);
    switch (@typeInfo(T)) {
        .void => {
            if (t.* != .void) return error.InvalidTypeInfo;
        },
        .bool => {
            if (t.* != .bool) return error.InvalidTypeInfo;
            try out.writeByte(@intFromBool(val));
            offset += 1;
        },
        .int => |it| {
            if (t.* != .int or it.signedness != t.int.signedness or it.bits != t.int.bits) return error.InvalidTypeInfo;
            const w = try writeInt(out, @TypeOf(val), val, endian);
            std.debug.assert(((it.bits - 1) / 8 + 1) == w);
            offset += w;
        },
        .float => |it| {
            if (t.* != .float or it.bits != t.float.bits) return error.InvalidTypeInfo;
            const Out = std.meta.Int(.unsigned, it.bits);
            const w = try writeInt(out, Out, @bitCast(val), endian);
            std.debug.assert(((t.float.bits - 1) / 8 + 1) == w);
            offset += w;
        },
        .array => |it| {
            if (t.* != .array or t.*.array.len != it.len) return error.InvalidTypeInfo;
            if (endian == native_endian and ((@typeInfo(it.child) == .int and @typeInfo(it.child).int.bits % 8 == 0) or
                (@typeInfo(it.child) == .float and @typeInfo(it.child).float.bits % 8 == 0)))
            {
                const data = std.mem.sliceAsBytes(&val);
                try out.writeAll(data);
                offset += data.len;
            } else {
                for (val) |entry| offset = try writeObject(out, offset, pointers, entry, t.array.child, fs, endian);
            }
        },
        .vector => |it| {
            if (t.* != .array or t.*.array.len != it.len) return error.InvalidTypeInfo;
            const tmp: [it.len]it.child = val;
            if (endian == native_endian and ((@typeInfo(it.child) == .int and @typeInfo(it.child).int.bits % 8 == 0) or
                (@typeInfo(it.child) == .float and @typeInfo(it.child).float.bits % 8 == 0)))
            {
                const data = std.mem.sliceAsBytes(&tmp);
                try out.writeAll(data);
                offset += data.len;
            } else {
                for (tmp) |entry| offset = try writeObject(out, offset, pointers, entry, t.array.child, fs, endian);
            }
        },
        .pointer => |it| {
            const ptr = switch (it.size) {
                .one => if (t.* != .pointer) return error.InvalidTypeInfo else @as(*anyopaque, @ptrCast(@constCast(val))),
                .slice => if (t.* != .vector) return error.InvalidTypeInfo else @as(*anyopaque, @ptrCast(@constCast(val.ptr))),
                .many, .c => @compileError("Unsupported slice type"),
            };

            if (pointers.get(ptr)) |entry| {
                try out.writeByte(0);
                offset += 1;
                const w = try writeInt(out, u64, entry.offset, endian);
                offset += w;
            } else {
                try out.writeByte(1);
                offset += 1;
                try pointers.put(ptr, PointerOffset{
                    .ptr = ptr,
                    .offset = offset,
                });

                switch (it.size) {
                    .one => {
                        offset = try writeObject(out, offset, pointers, val.*, t.pointer.child, fs, endian);
                    },
                    .slice => {
                        offset += try writeInt(out, u64, @intCast(val.len), endian);
                        if (endian == native_endian and ((@typeInfo(it.child) == .int and @typeInfo(it.child).int.bits % 8 == 0) or
                            (@typeInfo(it.child) == .float and @typeInfo(it.child).float.bits % 8 == 0)))
                        {
                            const data = std.mem.sliceAsBytes(val);
                            try out.writeAll(data);
                            offset += data.len;
                        } else {
                            for (val) |entry| offset = try writeObject(out, offset, pointers, entry, t.vector.child, fs, endian);
                        }
                    },
                    else => unreachable,
                }
            }
        },
        .optional => |_| {
            if (t.* != .optional) return error.InvalidTypeInfo;
            try out.writeByte(if (val == null) 0 else 1);
            offset += 1;
            if (val) |entry| offset = try writeObject(out, offset, pointers, entry, t.optional.child, fs, endian);
        },
        .@"enum" => |it| {
            if (!(t.* == .@"enum" and t.@"enum".fields_len == it.fields.len and lb: {
                inline for (it.fields, 0..) |f2, i| {
                    const f1 = fs[t.@"enum".fields_idx + i].@"enum";
                    if (!std.mem.eql(u8, f1.name, f2.name) or f1.value != @as(u32, @intCast(f2.value))) break :lb false;
                }
                break :lb true;
            })) return error.InvalidTypeInfo;
            const e_val = @as(u32, @intFromEnum(val));
            offset += try writeInt(out, u32, e_val, endian);
        },
        .@"union" => |it| {
            if (!(it.tag_type != null and t.* == .@"union" and t.@"union".fields_len == it.fields.len and lb: {
                inline for (it.fields, 0..) |f2, i| {
                    const f1 = fs[t.@"union".fields_idx + i].@"union";
                    if (!std.mem.eql(u8, f1.name, f2.name)) break :lb false;
                }
                break :lb true;
            })) return error.InvalidTypeInfo;
            switch (val) {
                inline else => |u_val, tag| {
                    const u_idx = @as(u32, @intFromEnum(tag));
                    offset += try writeInt(out, u32, u_idx, endian);
                    offset = try writeObject(out, offset, pointers, u_val, fs[t.@"union".fields_idx + u_idx].@"union".type, fs, endian);
                },
            }
        },
        .@"struct" => |it| {
            if (!(t.* == .@"struct" and t.@"struct".fields_len == it.fields.len and lb: {
                inline for (it.fields, 0..) |f2, i| {
                    const f1 = fs[t.@"struct".fields_idx + i].@"struct";
                    if (!std.mem.eql(u8, f1.name, f2.name)) break :lb false;
                }
                break :lb true;
            })) return error.InvalidTypeInfo;

            inline for (it.fields, 0..) |field, i| {
                const f = fs[t.@"struct".fields_idx + i].@"struct";
                offset = try writeObject(out, offset, pointers, @field(val, field.name), f.type, fs, endian);
            }
        },
        else => @compileError("Unsupported type: " ++ @typeName(T)),
    }

    return offset;
}

fn readObject(
    in: anytype,
    offset_in: usize,
    allocator: anytype,
    pointers: anytype,
    T: type,
    t: *const Type,
    fs: []const Type.Field,
    endian: std.builtin.Endian,
) (@TypeOf(in).Error || error{ InvalidTypeInfo, InvalidFormat, EndOfStream } || @TypeOf(allocator).Error || UnPtr(@TypeOf(pointers)).Error)!struct { T, usize } {
    var offset: usize = offset_in;
    var result: T = undefined;

    // @compileLog("readObject", T, offset);
    switch (@typeInfo(T)) {
        .void => {
            if (t.* != .void) return error.InvalidTypeInfo;
        },
        .bool => {
            if (t.* != .bool) return error.InvalidTypeInfo;
            result = switch (try in.readByte()) {
                0 => false,
                1 => true,
                else => return error.InvalidFormat,
            };
            offset += 1;
        },
        .int => |it| {
            if (t.* != .int or it.signedness != t.int.signedness or it.bits != t.int.bits) return error.InvalidTypeInfo;
            const value, const r = try readInt(in, T, endian);
            result = value;
            std.debug.assert(((it.bits - 1) / 8 + 1) == r);
            offset += r;
        },
        .float => |it| {
            if (t.* != .float or t.float.bits != it.bits) return error.InvalidTypeInfo;
            const In = std.meta.Int(.unsigned, it.bits);
            const value, const r = try readInt(in, In, endian);
            result = @bitCast(value);
            std.debug.assert(((t.float.bits - 1) / 8 + 1) == r);
            offset += r;
        },
        .array => |it| {
            if (t.* != .array or t.*.array.len != it.len) return error.InvalidTypeInfo;
            var tmp: [it.len]it.child = undefined;
            if (endian == native_endian and ((@typeInfo(it.child) == .int and @typeInfo(it.child).int.bits % 8 == 0) or
                (@typeInfo(it.child) == .float and @typeInfo(it.child).float.bits % 8 == 0)))
            {
                offset += try in.read(std.mem.sliceAsBytes(&tmp));
            } else for (0..it.len) |i| {
                tmp[i], offset = try readObject(in, offset, allocator, pointers, it.child, t.array.child, fs, endian);
            }
            result = tmp;
        },
        .vector => |it| {
            if (t.* != .array) return error.InvalidTypeInfo;
            var tmp: [it.len]it.child = undefined;
            if (endian == native_endian and ((@typeInfo(it.child) == .int and @typeInfo(it.child).int.bits % 8 == 0) or
                (@typeInfo(it.child) == .float and @typeInfo(it.child).float.bits % 8 == 0)))
            {
                offset += try in.read(std.mem.sliceAsBytes(&tmp));
            } else for (0..it.len) |i| {
                tmp[i], offset = try readObject(in, offset, allocator, pointers, it.child, t.array.child, fs, endian);
            }
            result = tmp;
        },
        .pointer => |it| {
            const has_data = switch (try in.readByte()) {
                0 => false,
                1 => true,
                else => return error.InvalidFormat,
            };

            offset += 1;
            if (has_data) {
                switch (it.size) {
                    .one => {
                        if (t.* != .pointer) return error.InvalidTypeInfo;
                        const v = try allocator.create(it.child);
                        try pointers.put(offset, PointerOffset{
                            .ptr = @as(*anyopaque, @ptrCast(v)),
                            .len = @sizeOf(it.child),
                            .alignment = std.mem.Alignment.fromByteUnits(@alignOf(it.child)),
                            .offset = offset,
                        });
                        v.*, offset = try readObject(in, offset, allocator, pointers, it.child, t.pointer.child, fs, endian);
                        result = v;
                    },
                    .slice => {
                        if (t.* != .vector) return error.InvalidTypeInfo;
                        const len, const r = try readInt(in, u64, endian);
                        offset += r;
                        const tmp = try allocator.alloc(it.child, len);
                        try pointers.put(offset - r, PointerOffset{
                            .ptr = @as(*anyopaque, @ptrCast(tmp.ptr)),
                            .len = @sizeOf(it.child) * len,
                            .alignment = std.mem.Alignment.fromByteUnits(@alignOf(it.child)),
                            .offset = offset - r,
                        });
                        if (endian == native_endian and ((@typeInfo(it.child) == .int and @typeInfo(it.child).int.bits % 8 == 0) or
                            (@typeInfo(it.child) == .float and @typeInfo(it.child).float.bits % 8 == 0)))
                        {
                            offset += try in.read(std.mem.sliceAsBytes(tmp));
                        } else {
                            for (0..len) |i| {
                                tmp[i], offset = try readObject(in, offset, allocator, pointers, it.child, t.vector.child, fs, endian);
                            }
                        }
                        result = tmp;
                    },
                    .many, .c => @compileError("Unsupported slice type"),
                }
            } else {
                const offset_ptr, const r = try readInt(in, u64, endian);
                offset += r;

                if (pointers.get(offset_ptr)) |entry| {
                    switch (it.size) {
                        .one => result = @as(*it.child, @alignCast(@ptrCast(entry.ptr))),
                        .slice => {
                            result.ptr = @as([*]it.child, @alignCast(@ptrCast(entry.ptr)));
                            result.len = entry.len;
                        },
                        .many, .c => @compileError("Unsupported slice type"),
                    }
                } else {
                    return error.InvalidFormat;
                }
            }
        },
        .optional => |it| {
            if (t.* != .optional) return error.InvalidTypeInfo;
            const has_data = switch (try in.readByte()) {
                0 => false,
                1 => true,
                else => return error.InvalidFormat,
            };
            offset += 1;
            if (has_data) {
                result, offset = try readObject(in, offset, allocator, pointers, it.child, t.optional.child, fs, endian);
            } else {
                result = null;
            }
        },
        .@"enum" => |it| {
            if (!(t.* == .@"enum" and t.@"enum".fields_len == it.fields.len and lb: {
                inline for (it.fields, 0..) |f2, i| {
                    const f1 = fs[t.@"enum".fields_idx + i].@"enum";
                    if (!std.mem.eql(u8, f1.name, f2.name) or f1.value != @as(u32, @intCast(f2.value))) break :lb false;
                }
                break :lb true;
            })) {
                return error.InvalidTypeInfo;
            }
            const value, const r = try readInt(in, u32, endian);
            offset += r;
            result = std.meta.intToEnum(T, value) catch return error.InvalidFormat;
        },
        .@"union" => |it| {
            if (!(it.tag_type != null and t.* == .@"union" and t.@"union".fields_len == it.fields.len and lb: {
                inline for (it.fields, 0..) |f2, i| {
                    const f1 = fs[t.@"union".fields_idx + i].@"union";
                    if (!std.mem.eql(u8, f1.name, f2.name)) break :lb false;
                }
                break :lb true;
            })) {
                return error.InvalidTypeInfo;
            }

            const value, const r = try readInt(in, u32, endian);
            offset += r;
            const tag = std.meta.intToEnum(it.tag_type.?, value) catch return error.InvalidFormat;
            switch (tag) {
                inline else => |u_tag| {
                    if (tag == u_tag) {
                        const field_type = @TypeOf(@field(result, @tagName(u_tag)));
                        const f = fs[t.@"union".fields_idx + @intFromEnum(u_tag)].@"union";
                        const v, offset = try readObject(in, offset, allocator, pointers, field_type, f.type, fs, endian);
                        result = @unionInit(T, @tagName(u_tag), v);
                    }
                },
            }
        },
        .@"struct" => |it| {
            if (!(t.* == .@"struct" and t.@"struct".fields_len == it.fields.len and lb: {
                inline for (it.fields, 0..) |f2, i| {
                    const f1 = fs[t.@"struct".fields_idx + i].@"struct";
                    if (!std.mem.eql(u8, f1.name, f2.name)) break :lb false;
                }
                break :lb true;
            })) {
                return error.InvalidTypeInfo;
            }

            inline for (it.fields, 0..) |field, i| {
                const f = fs[t.@"struct".fields_idx + i].@"struct";
                @field(result, field.name), offset = try readObject(in, offset, allocator, pointers, field.type, f.type, fs, endian);
            }
        },
        else => @compileError("Unsupported type: " ++ @typeName(T)),
    }

    return .{ result, offset };
}

fn containPointers(comptime T: type) bool {
    comptime {
        switch (@typeInfo(T)) {
            .array => |it| return containPointers(it.child),
            .vector => |it| return containPointers(it.child),
            .optional => |it| return containPointers(it.child),
            .pointer => return true,
            .@"union" => |it| {
                for (it.fields) |field| if (containPointers(field.type)) return true;
            },
            .@"struct" => |it| {
                for (it.fields) |field| if (containPointers(field.type)) return true;
            },
            else => return false,
        }
    }
}

inline fn writeInt(out: anytype, comptime T: type, value: T, endian: std.builtin.Endian) @TypeOf(out).Error!usize {
    const Out = std.math.ByteAlignedInt(T);
    var bytes: [@divExact(@typeInfo(Out).int.bits, 8)]u8 = undefined;
    std.mem.writeInt(Out, &bytes, @intCast(value), endian);
    try out.writeAll(&bytes);
    return bytes.len;
}

test writeInt {
    {
        var buffer: [20]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buffer);
        const out = stream.writer();
        const size_l = try writeInt(out, u64, 0x8899AABBCCDDEEFF, .little);
        try testing.expectEqual(8, size_l);
        try testing.expectEqualSlices(u8, &.{ 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88 }, buffer[0..8]);

        const size_b = try writeInt(out, u64, 0x8899AABBCCDDEEFF, .big);
        try testing.expectEqual(8, size_b);
        try testing.expectEqualSlices(u8, &.{ 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF }, buffer[8..16]);
    }
    {
        var buffer: [20]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buffer);
        const out = stream.writer();
        const size_l = try writeInt(out, u61, 0x0099AABBCCDDEEFF, .little);
        try testing.expectEqual(8, size_l);
        try testing.expectEqualSlices(u8, &.{ 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x00 }, buffer[0..8]);

        const size_b = try writeInt(out, u61, 0x0099AABBCCDDEEFF, .big);
        try testing.expectEqual(8, size_b);
        try testing.expectEqualSlices(u8, &.{ 0x00, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF }, buffer[8..16]);
    }
}

inline fn readInt(in: anytype, comptime T: type, endian: std.builtin.Endian) (@TypeOf(in).Error || error{EndOfStream})!struct { T, usize } {
    const In = std.math.ByteAlignedInt(T);
    const bytes = try in.readBytesNoEof(@divExact(@typeInfo(In).int.bits, 8));
    return .{ @intCast(std.mem.readInt(In, &bytes, endian)), bytes.len };
}

test readInt {
    {
        var buffer: [20]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buffer);
        var stream_in = std.io.fixedBufferStream(&buffer);
        const out = stream.writer();
        const in = stream_in.reader();
        _ = try writeInt(out, u64, 0x8899AABBCCDDEEFF, .little);
        var int, const size_l = try readInt(in, u64, .little);
        try testing.expectEqual(8, size_l);
        try testing.expectEqual(0x8899AABBCCDDEEFF, int);

        _ = try writeInt(out, u64, 0x8899AABBCCDDEEFF, .big);
        int, const size_b = try readInt(in, u64, .big);
        try testing.expectEqual(8, size_b);
        try testing.expectEqual(0x8899AABBCCDDEEFF, int);
    }
    {
        var buffer: [20]u8 = undefined;
        var stream = std.io.fixedBufferStream(&buffer);
        var stream_in = std.io.fixedBufferStream(&buffer);
        const out = stream.writer();
        const in = stream_in.reader();
        _ = try writeInt(out, u61, 0x0099AABBCCDDEEFF, .little);
        var int, const size_l = try readInt(in, u61, .little);
        try testing.expectEqual(8, size_l);
        try testing.expectEqual(0x0099AABBCCDDEEFF, int);

        _ = try writeInt(out, u61, 0x0099AABBCCDDEEFF, .big);
        int, const size_b = try readInt(in, u61, .big);
        try testing.expectEqual(8, size_b);
        try testing.expectEqual(0x0099AABBCCDDEEFF, int);
    }
}

const NamedType = struct {
    name: []const u8 = undefined,
    buf: [32]u8 = undefined,
    type: *const Type,
    list_node: ListDoubly.Node = LIST_D_NODE_NULL,
};

pub fn prettyPrintType(
    allocator: anytype,
    t: *const Type,
    fields: []const Type.Field,
    writer: anytype,
    indent_depth: u32,
    indent: []const u8,
    linebreak: []const u8,
) !void {
    var stack = ListDoubly{};
    defer {
        var node = stack.first;
        while (node) |n| {
            node = n.next;
            allocator.destroy(@as(*NamedType, @fieldParentPtr("list_node", n)));
        }
    }
    const res = try prettyPrintTypeImpl(allocator, t, fields, &stack, writer, indent_depth, indent, linebreak);
    return res;
}

fn writeIndent(writer: anytype, indent: []const u8, indent_depth: u32) !void {
    var i: u32 = 0;
    while (i < indent_depth) : (i += 1) try writer.writeAll(indent);
}

fn prettyPrintTypeImpl(
    allocator: anytype,
    t: *const Type,
    fields: []const Type.Field,
    stack: *ListDoubly,
    writer: anytype,
    indent_depth: u32,
    indent: []const u8,
    linebreak: []const u8,
) !void {
    {
        var node = stack.first;
        while (node) |n| {
            const entry: *NamedType = @fieldParentPtr("list_node", n);
            if (entry.type == t) {
                try writer.print("{s}", .{entry.name});
                return;
            }
            node = n.next;
        }
    }

    switch (t.*) {
        .void => try writer.writeAll("void"),
        .bool => try writer.writeAll("bool"),
        .int => |it| {
            const sign = switch (it.signedness) {
                .signed => "i",
                .unsigned => "u",
            };
            try writer.print("{s}{}", .{ sign, it.bits });
        },
        .float => |it| try writer.print("f{}", .{it.bits}),
        .array => |it| {
            try writer.print("[{}]", .{it.len});
            try prettyPrintTypeImpl(allocator, it.child, fields, stack, writer, indent_depth, indent, linebreak);
        },
        .vector => |it| {
            try writer.writeAll("[]");
            try prettyPrintTypeImpl(allocator, it.child, fields, stack, writer, indent_depth, indent, linebreak);
        },
        .pointer => |it| {
            try writer.writeAll("*");
            try prettyPrintTypeImpl(allocator, it.child, fields, stack, writer, indent_depth, indent, linebreak);
        },
        .optional => |it| {
            try writer.writeAll("?");
            try prettyPrintTypeImpl(allocator, it.child, fields, stack, writer, indent_depth, indent, linebreak);
        },
        .@"enum" => |it| {
            try writer.print("enum {{{s}", .{linebreak});
            for (it.fields_idx..(it.fields_idx + it.fields_len)) |i| {
                const f = fields[i].@"enum";
                try writeIndent(writer, indent, indent_depth + 1);
                try writer.print("{s} = {},{s}", .{ f.name, f.value, linebreak });
            }
            try writeIndent(writer, indent, indent_depth);
            try writer.writeAll("}");
        },

        .@"struct" => |it| {
            var nt = try allocator.create(NamedType);
            nt.* = NamedType{ .name = &.{}, .type = t };
            nt.name = std.fmt.bufPrint(&nt.buf, "Struct_{}", .{stack.len}) catch unreachable;
            stack.append(&nt.list_node);
            try writer.print("struct {s} {{{s}", .{ nt.name, linebreak });
            for (it.fields_idx..(it.fields_idx + it.fields_len)) |i| {
                const f = fields[i].@"struct";
                try writeIndent(writer, indent, indent_depth + 1);
                try writer.print("{s}: ", .{f.name});
                try prettyPrintTypeImpl(allocator, f.type, fields, stack, writer, indent_depth + 1, indent, linebreak);
                try writer.print(",{s}", .{linebreak});
            }
            try writeIndent(writer, indent, indent_depth);
            try writer.writeAll("}");
        },

        .@"union" => |it| {
            var nt = try allocator.create(NamedType);
            nt.* = NamedType{ .name = &.{}, .type = t };
            nt.name = std.fmt.bufPrint(&nt.buf, "Union_{}", .{stack.len}) catch unreachable;
            stack.append(&nt.list_node);
            try writer.print("union {s} {{{s}", .{ nt.name, linebreak });

            for (it.fields_idx..(it.fields_idx + it.fields_len)) |i| {
                const f = fields[i].@"union";
                try writeIndent(writer, indent, indent_depth + 1);
                try writer.print("{s}: ", .{f.name});
                try prettyPrintTypeImpl(allocator, f.type, fields, stack, writer, indent_depth + 1, indent, linebreak);
                try writer.print(",{s}", .{linebreak});
            }
            try writeIndent(writer, indent, indent_depth);
            try writer.writeAll("}");
        },
    }
}

fn HashMapComptime(Key: type, Value: type) type {
    const allocator = ComptimeAllocator{};

    const Entry = struct {
        value: Value,
        key: Key,
        list_node: ListSingly.Node = LIST_S_NODE_NULL,
    };

    return struct {
        pub const Error = error{};

        pointers: ListSingly = ListSingly{},

        fn init(_: anytype) @This() {
            return .{ .pointers = ListSingly{} };
        }

        fn deinit(_: *@This()) void {}

        fn put(self: *@This(), k: Key, v: Value) !void {
            const n = try allocator.create(Entry);
            n.* = Entry{ .value = v, .key = k };
            self.pointers.prepend(&n.list_node);
        }

        fn get(self: *@This(), k: Key) ?Value {
            var node = self.pointers.first;
            while (node) |n| {
                const entry: *Entry = @fieldParentPtr("list_node", n);
                if (entry.key == k) return entry.value;
                node = n.next;
            }
            return null;
        }
    };
}

fn HashMapRuntime(Key: type, Value: type) type {
    return struct {
        pub const Error = std.mem.Allocator.Error;

        map: std.AutoHashMap(Key, Value),

        fn init(allocator: std.mem.Allocator) @This() {
            return .{ .map = std.AutoHashMap(Key, Value).init(allocator) };
        }

        fn deinit(self: *@This()) void {
            self.map.deinit();
            self.* = undefined;
        }

        fn put(self: *@This(), k: Key, v: Value) !void {
            try self.map.put(k, v);
        }

        fn get(self: *@This(), key: Key) ?Value {
            return self.map.get(key);
        }
    };
}

const ComptimeAllocator = struct {
    const Self = @This();
    pub const Error = std.mem.Allocator.Error;

    pub fn alloc(_: Self, comptime T: type, comptime n: usize) Error![]T {
        var buffer = [_]T{undefined} ** n;
        return &buffer;
    }

    pub fn create(self: Self, comptime T: type) Error!*T {
        var val = try self.alloc(T, 1);
        return &val[0];
    }

    pub fn free(_: Self, _: anytype) void {}

    pub fn destroy(_: Self, _: anytype) void {}
};

pub fn UnPtr(comptime T: type) type {
    const info = @typeInfo(T);
    if (info == .pointer) {
        return info.pointer.child;
    }
    return T;
}

const testing = std.testing;

const EverythingStruct = struct {
    f0: void = {},
    f1: u1 = 1,
    f2: u64 = 0xFFAABBCCAA334455,
    f3: u256 = ~@as(u256, 0) - 1,
    f4: f80 = 0.6969696,
    f5: bool = true,
    f6: *const u32 = &54,
    f7: []const u8 = &.{ 1, 2, 3, 4 },
    f8: []const u21 = &.{ 54545, 4545, 4545, 4545 },
    f9: @Vector(2, u64) = .{ 99999, 11111 },
    f10: union(enum) { a: void, b: bool, c: []u64 } = .{ .b = true },
    f11: enum { a, b, c, d } = .a,
    f12: [5]u4 = .{ 1, 2, 3, 4, 5 },
    f13: ?enum { x, y, z } = null,

    fn eql(a: @This(), b: @This()) bool {
        return a.f1 == b.f1 and a.f2 == b.f2 and a.f3 == b.f3 and a.f4 == b.f4 and a.f5 == b.f5 and
            a.f6.* == b.f6.* and std.mem.eql(u8, a.f7, b.f7) and std.mem.eql(u21, a.f8, b.f8) and std.meta.eql(a.f9, b.f9) and
            std.meta.eql(a.f10, b.f10) and a.f11 == b.f11 and std.meta.eql(a.f12, b.f12) and a.f13 == b.f13;
    }
};

test "comptime build type info" {
    @setEvalBranchQuota(100000);
    comptime {
        _, const fields, const n_types, const bytes_required = toTypeInfo(EverythingStruct);

        try testing.expectEqual(24, fields.len);
        try testing.expectEqual(21, n_types);
        try testing.expectEqual(214, bytes_required);
    }
}

const RecursiveNode = struct {
    next: ?*RecursiveNode,
    prev: ?*RecursiveNode,
    data: u64,
};

test "build type info for recursive struct" {
    comptime {
        _, const fields, _, _ = toTypeInfo(RecursiveNode);
        try testing.expectEqual(3, fields.len);
    }
}

test "write type info for recursive struct" {
    comptime {
        const t, const fields, const n_types, const br = toTypeInfo(RecursiveNode);
        try testing.expectEqual(3, fields.len);
        var buffer = [_]u8{0} ** (br * 2);
        var stream = std.io.fixedBufferStream(&buffer);
        const writer = stream.writer();
        var offsets1: [n_types]TypeOffset, var offsets1_idx: usize = .{ undefined, 0 };
        const s1 = writeTypeInfo(writer, 0, t, &fields, &offsets1, &offsets1_idx, .big);
        try testing.expectEqual(br, s1);
        var offsets2: [n_types]TypeOffset, var offsets2_idx: usize = .{ undefined, 0 };
        const s2 = writeTypeInfo(writer, 0, t, &fields, &offsets2, &offsets2_idx, .little);
        try testing.expectEqual(s1, s2);
        try testing.expectEqual(39, s1);
    }
}

test "use type info in runtime" {
    _, const fields, _, const br = comptime toTypeInfo(RecursiveNode);
    try testing.expectEqual(3, fields.len);
    try testing.expectEqual(39, br);
}

test "type equality checks" {
    @setEvalBranchQuota(100000);
    comptime {
        {
            const t1, const fields1, _, _ = toTypeInfo(RecursiveNode);
            const t2, const fields2, _, _ = toTypeInfo(RecursiveNode);
            try testing.expect(isTypesEqual(t1, &fields1, t2, &fields2));
        }
        {
            const t1, const fields1, _, _ = toTypeInfo(RecursiveNode);
            const t2, const fields2, _, _ = toTypeInfo(EverythingStruct);
            try testing.expect(!isTypesEqual(t1, &fields1, t2, &fields2));
        }
    }
    {
        const t1, const fields1, _, _ = comptime toTypeInfo(RecursiveNode);
        const t2, const fields2, _, _ = comptime toTypeInfo(RecursiveNode);
        try testing.expect(isTypesEqual(t1, &fields1, t2, &fields2));
    }
}

test "read/write type info comptime" {
    @setEvalBranchQuota(100000);
    comptime {
        const t, const fields, const n_types, _ = toTypeInfo(EverythingStruct);
        var buffer = [_]u8{0} ** 512;
        {
            var stream_w = std.io.fixedBufferStream(&buffer);
            const writer = stream_w.writer();
            var offsets1: [n_types]TypeOffset, var offsets1_idx: usize = .{ undefined, 0 };
            const s1 = writeTypeInfo(writer, 0, t, &fields, &offsets1, &offsets1_idx, .little);
            var offsets2: [n_types]TypeOffset, var offsets2_idx: usize = .{ undefined, 0 };
            const s2 = writeTypeInfo(writer, 0, t, &fields, &offsets2, &offsets2_idx, .big);

            try testing.expectEqual(s1, s2);
            try testing.expectEqual(214, s1);
        }
        {
            var stream_r = std.io.fixedBufferStream(&buffer);
            const reader = stream_r.reader();

            const ca = ComptimeAllocator{};
            var fields_r = [_]Type.Field{undefined} ** fields.len;
            var offsets_r = HashMapComptime(u32, *const Type){};
            var fields_idx: usize = 0;
            const t2, const s1 = try readTypeInfo(reader, 0, ca, &fields_r, &fields_idx, &offsets_r, .little);

            try testing.expectEqual(214, s1);
            try testing.expect(isTypesEqual(t, &fields, t2, &fields_r));
        }
    }
}

test "read/write type info runtime" {
    @setEvalBranchQuota(100000);
    const t, const fields, const n_types, _ = comptime toTypeInfo(EverythingStruct);
    const buffer = try testing.allocator.alloc(u8, 512);
    defer testing.allocator.free(buffer);
    {
        var stream_w = std.io.fixedBufferStream(buffer);
        const writer = stream_w.writer();
        var offsets1: [n_types]TypeOffset, var offsets1_idx: usize = .{ undefined, 0 };
        const s1 = try writeTypeInfo(writer, 0, t, &fields, &offsets1, &offsets1_idx, .little);

        var offsets2: [n_types]TypeOffset, var offsets2_idx: usize = .{ undefined, 0 };
        const s2 = try writeTypeInfo(writer, 0, t, &fields, &offsets2, &offsets2_idx, .big);

        try testing.expectEqual(s1, s2);
        try testing.expectEqual(214, s1);
    }

    {
        var stream_r = std.io.fixedBufferStream(buffer);
        const reader = stream_r.reader();

        var aa = std.heap.ArenaAllocator.init(testing.allocator);
        defer aa.deinit();
        const ra = aa.allocator();
        const fields_r = try ra.alloc(Type.Field, fields.len);
        var fields_idx: usize = 0;
        var offsets_r = HashMapRuntime(u32, *const Type).init(ra);
        defer offsets_r.deinit();
        try offsets_r.map.ensureUnusedCapacity(n_types);
        const t2, const s1 = try readTypeInfo(reader, 0, ra, fields_r, &fields_idx, &offsets_r, .little);

        try testing.expectEqual(214, s1);
        try testing.expect(isTypesEqual(t, &fields, t2, fields_r));
    }
}

test "read/write data comptime " {
    @setEvalBranchQuota(100000);
    comptime {
        const value = EverythingStruct{};
        const t, const fields, const n_types, const br = toTypeInfo(@TypeOf(value));
        var buffer = [_]u8{0} ** 4096;
        const ca = ComptimeAllocator{};

        var sw_1: usize = undefined;
        var sw_2: usize = undefined;
        { // write
            var stream_w = std.io.fixedBufferStream(&buffer);
            const writer = stream_w.writer();
            var offsets1: [n_types]TypeOffset, var offsets1_idx: usize = .{ undefined, 0 };
            const s1 = writeTypeInfo(writer, 0, t, &fields, &offsets1, &offsets1_idx, .little);
            var offsets2: [n_types]TypeOffset, var offsets2_idx: usize = .{ undefined, 0 };
            const s2 = writeTypeInfo(writer, 0, t, &fields, &offsets2, &offsets2_idx, .big);

            try testing.expectEqual(s1, br);
            try testing.expectEqual(s1, s2);
            try testing.expectEqual(214, s1);

            var pointers_l = HashMapComptime(*anyopaque, PointerOffset){};
            sw_1 = try writeObject(writer, 0, &pointers_l, value, t, &fields, .little);

            var pointers_b = HashMapComptime(*anyopaque, PointerOffset){};
            sw_2 = try writeObject(writer, 0, &pointers_b, value, t, &fields, .big);

            try testing.expectEqual(sw_1, sw_2);
        }
        { // read
            var stream_r = std.io.fixedBufferStream(&buffer);
            const reader = stream_r.reader();

            {
                var fields_r = [_]Type.Field{undefined} ** fields.len;
                var offsets_r = HashMapComptime(u32, *const Type){};
                var fields_idx: usize = 0;
                const t2, const s1 = try readTypeInfo(reader, 0, ca, &fields_r, &fields_idx, &offsets_r, .little);

                try testing.expectEqual(214, s1);
                try testing.expect(isTypesEqual(t, &fields, t2, &fields_r));
            }
            {
                var fields_r = [_]Type.Field{undefined} ** fields.len;
                var offsets_r = HashMapComptime(u32, *const Type){};
                var fields_idx: usize = 0;
                _, _ = try readTypeInfo(reader, 0, ca, &fields_r, &fields_idx, &offsets_r, .big);
            }

            var pointers_l = HashMapComptime(u64, PointerOffset){};
            const d1, const s3 = try readObject(reader, 0, ca, &pointers_l, EverythingStruct, t, &fields, .little);
            var pointers_b = HashMapComptime(u64, PointerOffset){};
            var d2, const s4 = try readObject(reader, 0, ca, &pointers_b, EverythingStruct, t, &fields, .big);

            try testing.expectEqual(s3, s4);
            try testing.expectEqual(sw_1, s4);
            try testing.expectEqual(sw_2, s4);

            try testing.expect(d1.eql(d2));
            try testing.expect(value.eql(d2));
            d2.f2 = 0;
            try testing.expect(!d1.eql(d2));
        }
    }
}

test "read/write data runtime" {
    @setEvalBranchQuota(100000);
    var aa = std.heap.ArenaAllocator.init(testing.allocator);
    defer aa.deinit();
    const allocator = aa.allocator();

    const value = EverythingStruct{};
    var buffer = [_]u8{0} ** 4096;

    { // write
        var stream_w = std.io.fixedBufferStream(&buffer);
        const writer = stream_w.writer();
        _ = try serializeEndian(allocator, writer, value, .little);
        _ = try serializeEndian(allocator, writer, value, .big);
    }
    { // read
        var stream_r = std.io.fixedBufferStream(&buffer);
        const reader = stream_r.reader();

        const d1, _ = try deserializeEndian(allocator, reader, EverythingStruct, .little);
        const d2, _ = try deserializeEndian(allocator, reader, EverythingStruct, .big);

        try testing.expect(d1.eql(d2));
        try testing.expect(value.eql(d2));
    }
}

test "read/write data comptime" {
    @setEvalBranchQuota(100000);
    comptime {
        const value = EverythingStruct{};
        var buffer = [_]u8{0} ** 4096;

        { // write
            var stream_w = std.io.fixedBufferStream(&buffer);
            const writer = stream_w.writer();
            _ = try serializeEndianComptime(writer, value, .little);
            _ = try serializeEndianComptime(writer, value, .big);
        }
        { // read
            var stream_r = std.io.fixedBufferStream(&buffer);
            const reader = stream_r.reader();

            const d1 = try deserializeEndianComptime(reader, EverythingStruct, .little);
            const d2 = try deserializeEndianComptime(reader, EverythingStruct, .big);

            try testing.expect(d1.eql(d2));
            try testing.expect(value.eql(d2));
        }
    }
}

test "read/write doubly/singly linked lists" {
    @setEvalBranchQuota(100000);
    var aa = std.heap.ArenaAllocator.init(testing.allocator);
    defer aa.deinit();
    const allocator = aa.allocator();
    var buffer = [_]u8{0} ** 4096;

    const L1 = std.DoublyLinkedList(u32);
    const L2 = std.SinglyLinkedList(u64);

    var value1 = L1{};
    var value2 = L2{};

    var nodes1 = [_]L1.Node{.{ .data = 99 }} ** 5;
    var nodes2 = [_]L2.Node{.{ .data = 333 }} ** 5;

    for (&nodes1) |*n| value1.append(n);
    for (&nodes2) |*n| value2.prepend(n);

    { // write
        var stream_w = std.io.fixedBufferStream(&buffer);
        const writer = stream_w.writer();
        _ = try serializeEndian(allocator, writer, value1, .little);
        _ = try serializeEndian(allocator, writer, value2, .big);
    }
    { // read
        var stream_r = std.io.fixedBufferStream(&buffer);
        const reader = stream_r.reader();

        const d1, var dd1 = try deserializeEndian(testing.allocator, reader, L1, .little);
        defer dd1.deinit(testing.allocator);
        const d2, var dd2 = try deserializeEndian(testing.allocator, reader, L2, .big);
        defer dd2.deinit(testing.allocator);

        try testing.expect(d1.len == value1.len);
        try testing.expect(d2.len() == value2.len());

        { // Doubly linked list
            var p1, var p2 = .{ value1.first, d1.first };
            while (p1 != null and p2 != null) {
                if (p1.?.data != p2.?.data) try testing.expect(false);
                p1, p2 = .{ p1.?.next, p2.?.next };
            }
            try testing.expect(p1 == null and p2 == null);
        }

        { // Singly linked list
            var p1, var p2 = .{ value2.first, d2.first };
            while (p1 != null and p2 != null) {
                if (p1.?.data != p2.?.data) try testing.expect(false);
                p1, p2 = .{ p1.?.next, p2.?.next };
            }
            try testing.expect(p1 == null and p2 == null);
        }
    }
}

test "print everything struct" {
    @setEvalBranchQuota(10000);
    const t, const fields, _, _ = comptime toTypeInfo(EverythingStruct);
    var buffer = [_]u8{0} ** 2048;
    var stream = std.io.fixedBufferStream(&buffer);
    try prettyPrintType(testing.allocator, t, &fields, stream.writer(), 0, "  ", "\n");
}

test "print recursive struct" {
    @setEvalBranchQuota(10000);
    const t, const fields, _, _ = comptime toTypeInfo(RecursiveNode);
    var buffer = [_]u8{0} ** 2048;
    var stream = std.io.fixedBufferStream(&buffer);
    try prettyPrintType(testing.allocator, t, &fields, stream.writer(), 0, "  ", "\n");
}
