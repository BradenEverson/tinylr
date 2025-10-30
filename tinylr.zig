//! Tiny Logistic Regression Runtime

const std = @import("std");

/// Creates a Logistic Regression Model structure for a given integer type
///
/// `T`:        Describes the type that weights and features hold. should be an int type,
///             in most cases it should probably also be signed for your hyperplane to do
///             anything significant
///
/// `features`: Describes the dimensionality of your features, this will be the size
///             of your weight array and all future `x` you bring in for inference
pub fn LogisticRegression(comptime T: type, comptime features: comptime_int) type {
    switch (@typeInfo(T)) {
        .int => {},
        else => @compileError("Logistic Regression type must be an Integer type"),
    }

    return struct {
        w: [features]T,
        bias: T,

        const Self = @This();

        pub fn init(w: [features]T, bias: T) Self {
            return Self{
                .w = w,
                .bias = bias,
            };
        }

        fn doublePercision() type {
            const info = @typeInfo(T);

            return @Type(.{
                .int = .{
                    .signedness = .signed,
                    .bits = info.int.bits * 2,
                },
            });
        }

        /// Performs an optimized version of logistic regression w/ threshold = 0.5
        /// This allows us to simplify the expression, if w dot x + bias >= 0, class 1 (true)
        /// Else, class 0 (false)
        pub fn infer(self: *const Self, x: [features]T) bool {
            var result: doublePercision() = 0;
            for (0..features) |i| {
                result += self.w[i] * x[i];
            }

            return result + self.bias >= 0;
        }
    };
}

test "construction" {
    const Model = LogisticRegression(i8, 2);
    const m: Model = Model.init(.{ 1, 2 }, 3);

    try std.testing.expectEqual(m.w, .{ 1, 2 });
    try std.testing.expectEqual(m.bias, 3);
}

test "inference" {
    const Model = LogisticRegression(i8, 2);
    const m: Model = Model.init(.{ 1, 2 }, 3);

    // Should be above hyperplane (class 1)
    const x1 = .{ 4, 5 };
    // Should be below hyperplane (class 0)
    const x2 = .{ -2, -2 };

    try std.testing.expect(m.infer(x1));
    try std.testing.expect(!m.infer(x2));
}
