//! Optional request progress callback. GPU work must be drained before an
//! engine returns cancellation to a caller that can release its buffers.
pub const RequestControl = struct {
    context: *anyopaque,
    poll: *const fn (*anyopaque) bool,

    pub fn alive(self: RequestControl) bool {
        return self.poll(self.context);
    }
};
