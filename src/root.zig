// SIMD-accelerated data processing helpers.

const std = @import("std");
const builtin = @import("builtin");

// VECTOR LENGTH UTILS

pub const suggestVectorLengthForCpu = std.simd.suggestVectorLengthForCpu;
pub const suggestVectorLength = std.simd.suggestVectorLength;

// This should be kept in line with suggestVectorLengthForCpu implementation
// Useful for determining the point at which it would be better to use a scalar implementation
pub fn minimumVectorLengthForCpu(comptime T: type, comptime cpu: std.Target.Cpu) ?comptime_int {
    const element_bit_size = @max(8, std.math.ceilPowerOfTwo(u16, @bitSizeOf(T)) catch unreachable);
    const vector_bit_size: u16 = blk: {
        if (cpu.arch.isX86()) {
            if (T == bool and std.Target.x86.featureSetHas(cpu.features, .prefer_mask_registers)) return 64;
            if (std.Target.x86.featureSetHasAny(cpu.features, .{ .mmx, .@"3dnow" })) break :blk 64;
            if (std.Target.x86.featureSetHasAny(cpu.features, .{ .prefer_128_bit, .sse })) break :blk 128;
            if (std.Target.x86.featureSetHasAny(cpu.features, .{ .prefer_256_bit, .avx2 })) break :blk 256;
            if (builtin.zig_backend != .stage2_x86_64 and std.Target.x86.featureSetHas(cpu.features, .avx512f)) break :blk 512;
        } else if (cpu.arch.isMIPS()) {
            // TODO: Test MIPS capability to handle bigger vectors
            //       In theory MDMX and by extension mips3d have 32 registers of 64 bits which can use in parallel
            //       for multiple processing, but I don't know what's optimal here, if using
            //       the 2048 bits or using just 64 per vector or something in between
            if (std.Target.mips.featureSetHas(cpu.features, std.Target.mips.Feature.mips3d)) break :blk 64;
            if (std.Target.mips.featureSetHas(cpu.features, .msa)) break :blk 128;
        } else break :blk suggestVectorLengthForCpu(T, cpu) orelse return null;
    };
    if (vector_bit_size <= element_bit_size) return null;

    return @divExact(vector_bit_size, element_bit_size);
}

pub inline fn minimumVectorLength(comptime T: type) ?comptime_int {
    return minimumVectorLengthForCpu(T, builtin.cpu);
}

// SIMD CHUNK PROCESSOR

pub fn Processor(
    // zig fmt: off
    comptime ItemT: type, comptime ArgsT: type,
    comptime ErrorT: type, comptime StateT: type, comptime init_state: StateT,
    comptime scalar_fn: fn (ItemT, ArgsT, *StateT) callconv(.@"inline") ErrorT!void,
    comptime simd_fn: fn (comptime type, anytype, ArgsT, *StateT) callconv(.@"inline") ErrorT!void,
    // zig fmt: on
) type {
    // use suggested vector length instead of largest possible
    const vec_len = std.simd.suggestVectorLength(ItemT) orelse 0;

    return struct {
        pub inline fn process(slice: []const ItemT, args: ArgsT) ErrorT!StateT {
            var state: StateT = init_state;
            var items = slice;

            // no simd support
            if (comptime vec_len == 0) {
                for (items) |item|
                    try scalar_fn(item, args, &state);

                return state;
            }

            // handle largest vec length first

            {
                const T = @Vector(vec_len, ItemT);

                const processable = items.len / vec_len;
                const items_processable = processable * vec_len;

                for (0..processable) |_| {
                    const vec: T = items[0..vec_len].*;
                    try simd_fn(T, vec, args, &state);

                    items.ptr += vec_len;
                }

                items.len -= items_processable;
            }

            // as vec length trickles down to handle the remaining items,
            // we'll only have enough left to fit into a single vector.
            // checks are the same as above implementation
            // but we skip a lot of unnecessary steps

            comptime var vlen = vec_len / 2;
            inline while (vlen > 1) {
                if (items.len >= vlen) {
                    const T = @Vector(vlen, u8);

                    const vec: T = items[0..vlen].*;
                    try simd_fn(T, vec, args, &state);

                    items.len -= vlen;
                    items.ptr += vlen;
                }

                if (items.len == 0) return;

                vlen /= 2;
            }

            // we'll only have a single item left to process at this point

            try scalar_fn(items[0], args, &state);
            return state;
        }
    };
}
