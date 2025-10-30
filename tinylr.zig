//! Tiny Logistic Regression Runtime

const std = @import("std");

pub fn LrModel(comptime T: type, comptime features: comptime_int) type {
    return struct {
        w: [features]T,
        w0: T,

        const Self = @This();

        pub fn init(w: [features]T, w0: T) Self {
            return Self{
                .w = w,
                .w0 = w0,
            };
        }

        /// Performs an optimized version of logistic regression w/ threshold = 0.5
        /// This allows us to simplify the expression, if w dot (x - w0) >= 0, class 1 (true)
        /// Else, class 0 (false)
        pub fn infer(self: *const Self, x: [features]T) bool {
            _ = self;
            _ = x;
            false;
        }
    };
}

test "construction" {
    const Model = LrModel(u8, 2);
    const m: Model = Model.init(.{ 1, 2 }, 3);

    try std.testing.expectEqual(m.w, .{ 1, 2 });
    try std.testing.expectEqual(m.w0, 3);
}
