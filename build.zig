const std = @import("std");

pub fn build(b: *std.Build) void {
    // OPTIONS

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // MODULE

    const options = b.addOptions();
    const options_module = options.createModule(); // reserved
    const simdutil = b.addModule("simdutil", .{
        .root_source_file = b.path("src/root.zig"),
        .imports = &.{.{ .name = "options", .module = options_module }},
    });

    // TESTS

    const tests = b.addTest(.{
        .root_source_file = b.path("src/tests.zig"),
        .target = target,
        .optimize = optimize,
    });

    tests.linkLibC();
    tests.root_module.addImport("simdutil", simdutil);

    const run_tests = b.addRunArtifact(tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
