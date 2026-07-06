//! PSTD command and queue provider contracts.

use std::ops::Range;

/// Provider contract for run-loop command and queue operations.
pub(in crate::pstd_gpu) trait PstdCommandProvider {
    /// Provider-owned buffer type.
    type Buffer;

    /// Provider-owned command encoder type.
    type Encoder;

    /// Provider-owned compute pass type.
    type ComputePass<'pass>;

    /// Clear a device buffer range and submit the command.
    fn clear_buffer(&self, buffer: &Self::Buffer, size_bytes: u64, label: &'static str);

    /// Copy one device buffer range into another and submit the command.
    fn copy_buffer(
        &self,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        size_bytes: u64,
        label: &'static str,
    );

    /// Read a mapped staging buffer range into host scalars.
    fn read_mapped<T>(&self, buffer: &Self::Buffer, size_bytes: u64) -> Vec<T>
    where
        T: bytemuck::Pod;

    /// Write host scalar data into a provider-owned buffer.
    fn write_buffer<T>(&self, buffer: &Self::Buffer, offset_bytes: u64, data: &[T])
    where
        T: bytemuck::Pod;

    /// Create a command encoder, let the caller encode provider-native work,
    /// then submit the completed command buffer.
    fn submit_encoder<F>(&self, label: &'static str, encode: F)
    where
        F: FnOnce(&mut Self::Encoder);

    /// Submit one provider-native compute pass in a command buffer.
    fn submit_compute_pass<F>(
        &self,
        encoder_label: &'static str,
        pass_label: &'static str,
        encode: F,
    ) where
        F: for<'pass> FnOnce(&mut Self::ComputePass<'pass>);

    /// Submit one command buffer containing one provider-native compute pass
    /// for each item in `items`.
    fn submit_compute_passes<F>(
        &self,
        encoder_label: &'static str,
        pass_label: &'static str,
        items: Range<usize>,
        encode: F,
    ) where
        F: for<'pass> FnMut(usize, &mut Self::ComputePass<'pass>);

    /// Wait for provider work to complete.
    fn poll_wait(&self);
}

/// WGPU PSTD command and queue provider.
pub(in crate::pstd_gpu) struct WgpuPstdCommandProvider<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
}

impl<'a> WgpuPstdCommandProvider<'a> {
    /// Create a WGPU command provider for PSTD run-loop operations.
    #[must_use]
    pub(in crate::pstd_gpu) const fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue) -> Self {
        Self { device, queue }
    }
}

impl PstdCommandProvider for WgpuPstdCommandProvider<'_> {
    type Buffer = wgpu::Buffer;
    type Encoder = wgpu::CommandEncoder;
    type ComputePass<'pass> = wgpu::ComputePass<'pass>;

    fn clear_buffer(&self, buffer: &Self::Buffer, size_bytes: u64, label: &'static str) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encoder.clear_buffer(buffer, 0, Some(size_bytes));
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn copy_buffer(
        &self,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        size_bytes: u64,
        label: &'static str,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encoder.copy_buffer_to_buffer(source, 0, destination, 0, size_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn read_mapped<T>(&self, buffer: &Self::Buffer, size_bytes: u64) -> Vec<T>
    where
        T: bytemuck::Pod,
    {
        let slice = buffer.slice(..size_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.poll_wait();
        rx.recv()
            .expect("invariant: WGPU map callback must run after provider poll_wait")
            .expect("PSTD staging buffer map failed");
        let mapped = slice.get_mapped_range();
        let result = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        buffer.unmap();
        result
    }

    fn write_buffer<T>(&self, buffer: &Self::Buffer, offset_bytes: u64, data: &[T])
    where
        T: bytemuck::Pod,
    {
        self.queue
            .write_buffer(buffer, offset_bytes, bytemuck::cast_slice(data));
    }

    fn submit_encoder<F>(&self, label: &'static str, encode: F)
    where
        F: FnOnce(&mut Self::Encoder),
    {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        encode(&mut encoder);
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    fn submit_compute_pass<F>(
        &self,
        encoder_label: &'static str,
        pass_label: &'static str,
        encode: F,
    ) where
        F: for<'pass> FnOnce(&mut Self::ComputePass<'pass>),
    {
        self.submit_encoder(encoder_label, |encoder| {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(pass_label),
                timestamp_writes: None,
            });
            encode(&mut cpass);
        });
    }

    fn submit_compute_passes<F>(
        &self,
        encoder_label: &'static str,
        pass_label: &'static str,
        items: Range<usize>,
        mut encode: F,
    ) where
        F: for<'pass> FnMut(usize, &mut Self::ComputePass<'pass>),
    {
        self.submit_encoder(encoder_label, |encoder| {
            for item in items {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(pass_label),
                    timestamp_writes: None,
                });
                encode(item, &mut cpass);
            }
        });
    }

    fn poll_wait(&self) {
        let _ = self.device.poll(wgpu::PollType::Wait);
    }
}

#[cfg(test)]
mod tests {
    use super::{PstdCommandProvider, WgpuPstdCommandProvider};

    #[test]
    fn pstd_command_provider_is_generic_over_provider_trait() {
        fn assert_provider<P>()
        where
            P: PstdCommandProvider + 'static,
        {
            let _ = core::mem::size_of::<P::Buffer>();
            let _ = core::mem::size_of::<P::Encoder>();
            let _ = core::mem::size_of::<P::ComputePass<'static>>();
        }

        assert_provider::<WgpuPstdCommandProvider<'static>>();
    }
}
