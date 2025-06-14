const std = @import("std");

/// Returns an error message if `Target` does not implement `Interface`, otherwise null.
pub fn checkImplementsMessage(comptime Interface: type, comptime Target: type) ?[]const u8 {
    comptime {
        const interface_info = @typeInfo(Interface);
        const target_info = @typeInfo(Target);

        if (interface_info != .@"struct" or target_info != .@"struct") {
            return "Both Interface and Target must be struct types";
        }

        const interface_fields = interface_info.@"struct".fields;
        const interface_decls = interface_info.@"struct".decls;

        for (interface_fields) |field| {
            if (!@hasField(Target, field.name)) {
                return "Target type '" ++ @typeName(Target) ++ "' missing required field '" ++ field.name ++ "'";
            }

            const target_field = std.meta.fieldInfo(Target, @field(std.meta.FieldEnum(Target), field.name));

            if (field.type != target_field.type) {
                return "Field '" ++ field.name ++ "' type mismatch: expected '" ++
                    @typeName(field.type) ++ "', found '" ++ @typeName(target_field.type) ++ "'";
            }
        }

        for (interface_decls) |decl| {
            if (!@hasDecl(Target, decl.name)) {
                return "Target type '" ++ @typeName(Target) ++ "' missing required declaration '" ++ decl.name ++ "'";
            }

            const interface_decl = @field(Interface, decl.name);
            const target_decl = @field(Target, decl.name);
            const interface_decl_type = @TypeOf(interface_decl);
            const target_decl_type = @TypeOf(target_decl);

            if (@typeInfo(interface_decl_type) == .@"fn" and @typeInfo(target_decl_type) == .@"fn") {
                const interface_fn = @typeInfo(interface_decl_type).@"fn";
                const target_fn = @typeInfo(target_decl_type).@"fn";

                if (interface_fn.params.len != target_fn.params.len) {
                    return "Function '" ++ decl.name ++ "' parameter count mismatch";
                }

                if (interface_fn.return_type) |i_rt| {
                    if (target_fn.return_type) |t_rt| {
                        if (i_rt != t_rt) {
                            return "Function '" ++ decl.name ++ "' return type mismatch; Type " ++
                                @typeName(t_rt) ++ " does not satisfy " ++ @typeName(i_rt);
                        }
                    } else {
                        return "Function '" ++ decl.name ++ "' return type mismatch";
                    }
                } else {
                    if (target_fn.return_type) |_| {
                        return "Function '" ++ decl.name ++ "' return type mismatch";
                    }
                }

                for (interface_fn.params, 0..) |param, i| {
                    if (param.type) |pi| {
                        if (target_fn.params[i].type) |pt| {
                            if (i == 0) {
                                const info_i = @typeInfo(pi);
                                const info_t = @typeInfo(pt);
                                const is_self_iface = pi == Interface or (info_i == .pointer and info_i.pointer.child == Interface);
                                const is_self_target = pt == Target or (info_t == .pointer and info_t.pointer.child == Target);

                                if (is_self_iface and is_self_target) {
                                    // *const Interface <-> *const Target
                                    // *Interface <-> *const Target
                                    // Interface <-> *const Target
                                    // *const Interface <-> Target
                                    // *Interface <-> Target
                                    // Interface <-> Target
                                    // *Interface <-> *Target
                                    if ((pt == Target and pi == Interface) or pt == Target or (info_i == .pointer and !info_i.pointer.is_const) or (info_t == .pointer and info_t.pointer.is_const)) {
                                        continue;
                                    }
                                    // *const Interface <-> *Target
                                    // Interface <-> *Target
                                    return "Function '" ++ decl.name ++ "' parameter " ++
                                        std.fmt.comptimePrint("{d}", .{i}) ++ " const mismatch";
                                }
                            }

                            if (pi == pt) continue;
                        }
                    } else {
                        if (target_fn.return_type == null) continue;
                    }

                    return "Function '" ++ decl.name ++ "' parameter " ++
                        std.fmt.comptimePrint("{d}", .{i}) ++ " type mismatch";
                }
            } else if (interface_decl_type != target_decl_type) {
                return "Declaration '" ++ decl.name ++ "' type mismatch: expected '" ++
                    @typeName(interface_decl_type) ++ "', found '" ++ @typeName(target_decl_type) ++ "'";
            }
        }

        return null;
    }
}

pub fn checkImplementsGuard(comptime Interface: type, comptime Target: type) void {
    comptime {
        if (checkImplementsMessage(Interface, Target)) |error_msg| {
            @compileError(error_msg);
        }
    }
}

pub fn checkImplements(comptime Interface: type, comptime Target: type) bool {
    comptime {
        return checkImplementsMessage(Interface, Target) == null;
    }
}

/// Removes one level of pointer indirection from a type, if it is a pointer
pub fn UnPtr(comptime T: type) type {
    const info = @typeInfo(T);
    if (info == .pointer) {
        return info.pointer.child;
    }
    return T;
}

test "CheckIfaceValidationPass" {
    const Writer = struct {
        buffer: []u8,

        pub fn write(_: *@This(), _: []const u8) usize {
            unreachable;
        }

        pub fn flush(_: *@This()) void {
            unreachable;
        }
    };

    const FileWriter = struct {
        buffer: []u8,
        file_handle: i32,

        pub fn write(_: *@This(), _: []const u8) usize {
            unreachable;
        }

        pub fn flush(_: *@This()) void {
            unreachable;
        }

        pub fn close(_: *@This()) void {
            unreachable;
        }
    };
    const result = comptime checkImplements(Writer, FileWriter);
    try std.testing.expect(result);
}

test "CheckIfaceValidationFail" {
    const Writer = struct {
        buffer: []u8,

        pub fn write(_: *@This(), _: []const u8) !usize {
            unreachable;
        }

        pub fn flush(_: *@This()) !void {
            unreachable;
        }
    };

    const BadWriter = struct {
        buffer: []u8,
        pub fn write(_: *@This(), _: []const u8) !usize {
            unreachable;
        }
    };

    const result = comptime checkImplements(Writer, BadWriter);
    try std.testing.expect(!result);
}
