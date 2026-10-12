#include <opencv2/opencv.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr const char* kSharedMemoryName = "/primus_cam";
constexpr std::size_t kHeaderBytes = 64;
constexpr int kRequestedWidth = 1920;
constexpr int kRequestedHeight = 1080;

void publish_frame(std::uint8_t* memory, const cv::Mat& bgr, cv::Mat& rgb,
                   std::uint64_t timestamp_ns, std::uint64_t fps) {
    auto* header = reinterpret_cast<std::uint64_t*>(memory);
    const std::uint64_t current = __atomic_load_n(&header[0], __ATOMIC_RELAXED);
    const std::uint64_t even_sequence = (current & 1) ? current + 1 : current;

    __atomic_store_n(&header[0], even_sequence + 1, __ATOMIC_RELEASE);
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    header[3] = timestamp_ns;
    header[4] = fps;
    __atomic_store_n(&header[0], even_sequence + 2, __ATOMIC_RELEASE);
}
}  // namespace

int main() {
    cv::VideoCapture camera;
    if (!camera.open(0, cv::CAP_V4L2) && !camera.open(0)) {
        std::fprintf(stderr, "[cam] Could not open camera 0 via V4L2 or auto-detect\n");
        return 1;
    }

    camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    camera.set(cv::CAP_PROP_AUTO_EXPOSURE, 3.0);
    camera.set(cv::CAP_PROP_FRAME_WIDTH, kRequestedWidth);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, kRequestedHeight);
    camera.set(cv::CAP_PROP_FPS, 30);

    cv::Mat bgr;
    if (!camera.read(bgr) || bgr.empty()) {
        std::fprintf(stderr, "[cam] Camera opened but returned no frame\n");
        return 1;
    }

    const int width = bgr.cols;
    const int height = bgr.rows;
    const std::size_t frame_bytes =
        static_cast<std::size_t>(width) * height * 3;
    const std::size_t mapping_bytes = kHeaderBytes + frame_bytes;

    const int fd = shm_open(kSharedMemoryName, O_CREAT | O_RDWR, 0660);
    if (fd < 0) {
        std::perror("[cam] shm_open");
        return 1;
    }
    if (ftruncate(fd, static_cast<off_t>(mapping_bytes)) != 0) {
        std::perror("[cam] ftruncate");
        close(fd);
        return 1;
    }

    auto* memory = static_cast<std::uint8_t*>(
        mmap(nullptr, mapping_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);
    if (memory == MAP_FAILED) {
        std::perror("[cam] mmap");
        return 1;
    }

    auto* header = reinterpret_cast<std::uint64_t*>(memory);
    header[0] = 0;
    header[1] = static_cast<std::uint64_t>(width);
    header[2] = static_cast<std::uint64_t>(height);
    header[3] = 0;
    header[4] = 0;
    const auto fourcc = static_cast<std::uint32_t>(
        camera.get(cv::CAP_PROP_FOURCC));
    std::printf("[cam] %dx%d via %s, %.1f fps, auto-exposure=%.2f, fourcc=%c%c%c%c\n",
                width, height, camera.getBackendName().c_str(),
                camera.get(cv::CAP_PROP_FPS),
                camera.get(cv::CAP_PROP_AUTO_EXPOSURE),
                static_cast<char>(fourcc & 0xff),
                static_cast<char>((fourcc >> 8) & 0xff),
                static_cast<char>((fourcc >> 16) & 0xff),
                static_cast<char>((fourcc >> 24) & 0xff));
    if (width != kRequestedWidth || height != kRequestedHeight) {
        std::fprintf(stderr, "[cam] Camera did not accept requested 1920x1080 mode\n");
    }

    cv::Mat rgb(height, width, CV_8UC3, memory + kHeaderBytes);
    std::uint64_t frames = 0;
    std::uint64_t published_fps = 0;
    std::chrono::nanoseconds read_time{0};
    std::chrono::nanoseconds publish_time{0};
    auto fps_window_start = std::chrono::steady_clock::now();

    while (true) {
        const auto read_start = std::chrono::steady_clock::now();
        if (!camera.read(bgr) || bgr.empty()) {
            break;
        }
        read_time += std::chrono::steady_clock::now() - read_start;
        if (bgr.cols != width || bgr.rows != height) {
            std::fprintf(stderr, "[cam] Camera frame size changed unexpectedly\n");
            break;
        }

        ++frames;
        const auto timestamp_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count());
        const auto publish_start = std::chrono::steady_clock::now();
        publish_frame(memory, bgr, rgb, timestamp_ns, published_fps);
        publish_time += std::chrono::steady_clock::now() - publish_start;

        const auto now = std::chrono::steady_clock::now();
        const auto elapsed =
            std::chrono::duration<double>(now - fps_window_start).count();
        if (elapsed >= 1.0) {
            published_fps = static_cast<std::uint64_t>(frames / elapsed);
            const double average_read_ms =
                std::chrono::duration<double, std::milli>(read_time).count()
                / frames;
            const double average_publish_ms =
                std::chrono::duration<double, std::milli>(publish_time).count()
                / frames;
            frames = 0;
            read_time = std::chrono::nanoseconds{0};
            publish_time = std::chrono::nanoseconds{0};
            fps_window_start = now;
            std::printf("[cam] capture %llu fps, read %.1f ms/frame, publish %.1f ms/frame\n",
                        static_cast<unsigned long long>(published_fps),
                        average_read_ms, average_publish_ms);
            std::fflush(stdout);
        }
    }

    munmap(memory, mapping_bytes);
    camera.release();
    return 0;
}
