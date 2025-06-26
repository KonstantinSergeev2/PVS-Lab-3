#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstdint>

// Последовательное суммирование массива
double sequential_sum(const std::vector<double>& array) {
    double sum = 0.0;
    for (double v : array) sum += v;
    return sum;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Общее число процессов

    uint64_t N = 0;
    int desired_procs = 0;

    if (rank == 0) {
        // Ввод размера массива
        std::cout << "Array size (N > 100000): ";
        std::cin >> N;
        while (N <= 100000) {
            std::cout << "Must be > 100000: ";
            std::cin >> N;
        }
        // Ввод желаемого числа процессов
        std::cout << "Desired number of processes: ";
        std::cin >> desired_procs;
    }

    // Передаём N и desired_procs всем процессам
    MPI_Bcast(&N, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&desired_procs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Проверка на совпадение желаемого и фактического числа процессов
    if (rank == 0 && desired_procs != size) {
        std::cout << "[Warning] Requested " << desired_procs
            << ", but running on " << size << " processes.\n";
    }

    // Генерация массива на процессе 0
    std::vector<double> array;
    if (rank == 0) {
        array.resize(N);
        std::mt19937_64 rng(static_cast<unsigned>(time(nullptr)));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        for (uint64_t i = 0; i < N; ++i) {
            array[i] = dist(rng);
        }
    }

    // Последовательное суммирование (только процесс 0)
    double seq_sum = 0.0, seq_time = 0.0;
    if (rank == 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        seq_sum = sequential_sum(array);
        auto t2 = std::chrono::high_resolution_clock::now();
        seq_time = std::chrono::duration<double>(t2 - t1).count();
    }

    // Разбиение массива между процессами
    uint64_t base = N / size;
    uint64_t rem = N % size;
    std::vector<int> sendcounts(size), displs(size);
    uint64_t offset = 0;
    for (int r = 0; r < size; ++r) {
        uint64_t cnt = base + (r < static_cast<int>(rem) ? 1 : 0);
        sendcounts[r] = static_cast<int>(cnt);
        displs[r] = static_cast<int>(offset);
        offset += cnt;
    }

    // Локальный буфер для каждого процесса
    int local_n = sendcounts[rank];
    std::vector<double> local_array(local_n);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Распределение данных
    MPI_Scatterv(
        rank == 0 ? array.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        local_array.data(),
        local_n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // Локальная сумма
    double local_sum = 0.0;
    for (double v : local_array) local_sum += v;

    // Сбор общей суммы
    double par_sum = 0.0;
    MPI_Reduce(&local_sum, &par_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double par_time = MPI_Wtime() - t_start;

    // Вывод результатов
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Sequential: time = " << seq_time << " s, sum = " << seq_sum << "\n";
        std::cout << "Parallel  : time = " << par_time << " s, sum = " << par_sum << "\n";
    }

    MPI_Finalize();
    return 0;
}
