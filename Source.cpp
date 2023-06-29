#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>

class ThreadPool {
public:
    explicit ThreadPool(size_t threadCount) : stop(false) {
        for (size_t i = 0; i < threadCount; ++i) {
            threads.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty())
                            return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
                });
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& thread : threads)
            thread.join();
    }

    template <typename Function, typename... Args>
    auto push_task(Function&& f, Args&&... args) -> std::future<typename std::result_of<Function(Args...)>::type> {
        using ReturnType = typename std::result_of<Function(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<ReturnType()>>(std::bind(std::forward<Function>(f), std::forward<Args>(args)...));
        std::future<ReturnType> result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return result;
    }

private:
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

template<typename T>
void quicksort(std::vector<T>& array, int left, int right) {
    if (left >= right)
        return;

    int pivot = partition(array, left, right);

    quicksort(array, left, pivot - 1);
    quicksort(array, pivot + 1, right);
}

template<typename T>
int partition(std::vector<T>& array, int left, int right) {
    T pivot = array[right];
    int i = left - 1;

    for (int j = left; j <= right - 1; ++j) {
        if (array[j] < pivot) {
            ++i;
            std::swap(array[i], array[j]);
        }
    }

    std::swap(array[i + 1], array[right]);
    return i + 1;
}

template<typename T>
void parallel_quicksort(std::vector<T>& array, int left, int right, ThreadPool& pool, std::shared_ptr<std::promise<void>> promise) {
    if (left >= right)
        return;

    int pivot = partition(array, left, right);

    if (right - left > 10000) {
        auto f = pool.push_task(parallel_quicksort<T>, std::ref(array), left, pivot - 1, std::ref(pool), promise);
        parallel_quicksort(array, pivot + 1, right, pool, promise);
        f.wait();
    }
    else {
        parallel_quicksort(array, left, pivot - 1, pool, promise);
        parallel_quicksort(array, pivot + 1, right, pool, promise);
    }

    if (promise) {
        std::lock_guard<std::mutex> lock(promise->mutex);
        promise->count--;
        if (promise->count == 0)
            promise->condition.notify_one();
    }
}

template<typename T>
void parallel_quicksort(std::vector<T>& array, ThreadPool& pool) {
    if (array.empty())
        return;

    std::shared_ptr<std::promise<void>> promise = nullptr;
    std::future<void> future;

    if (array.size() > 100000) {
        promise = std::make_shared<std::promise<void>>();
        future = promise->get_future();
        promise->count = 2;
    }

    parallel_quicksort(array, 0, array.size() - 1, pool, promise);

    if (future.valid())
        future.wait();
}

int main() {
    std::vector<int> numbers(1000000);
    std::iota(numbers.begin(), numbers.end(), 1);

    std::random_shuffle(numbers.begin(), numbers.end());

    ThreadPool pool(std::thread::hardware_concurrency());

    auto start = std::chrono::steady_clock::now();

    parallel_quicksort(numbers, pool);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Parallel Quicksort took " << elapsed_seconds.count() << " seconds.\n";

    return 0;
}
