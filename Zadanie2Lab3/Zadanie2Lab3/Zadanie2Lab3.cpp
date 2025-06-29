#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <ctime>
#include <fstream>
#include <algorithm>

// Последовательная пузырьковая сортировка.
void sequential_bubble_sort(std::vector<double>& array) {
    for (size_t i = 0; i < array.size(); ++i) {
        for (size_t j = 0; j + 1 < array.size() - i; ++j) {
            if (array[j] > array[j + 1]) {
                std::swap(array[j], array[j + 1]);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int proc, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Чтение размера массива на процессе 0.
    size_t N = 0;
    if (proc == 0) {
        if (argc >= 2) {
            N = static_cast<size_t>(std::atoll(argv[1]));
            if (N <= 100000) {
                std::cerr << "Error: Array size must be greater than 100000.\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else {
            std::cout << "Enter array size (must be greater than 100000): ";
            std::cin >> N;
            if (N <= 100000) {
                std::cerr << "Error: Array size must be greater than 100000.\n";
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }
    }

    // Рассылка размера массива всем процессам.
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Генерация полного массива на процессе 0.
    std::vector<double> full_array;
    if (proc == 0) {
        full_array.resize(N);
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        for (size_t i = 0; i < N; ++i) {
            full_array[i] = static_cast<double>(std::rand()) / RAND_MAX;
        }
    }

    double seq_time = 0.0, par_time = 0.0;

    // Вычисление sendcounts и displs для Scatterv/Gatherv
    std::vector<int> sendcounts(num_procs), displs(num_procs);
    size_t base = N / num_procs, rem = N % num_procs, ofs = 0;
    for (int p = 0; p < num_procs; ++p) {
        size_t cnt = base + (p < static_cast<int>(rem) ? 1 : 0);
        sendcounts[p] = static_cast<int>(cnt);
        displs[p] = static_cast<int>(ofs);
        ofs += cnt;
    }

    // Локальный буфер.
    std::vector<double> local_array(sendcounts[proc]);
    std::vector<double> seq_array;
    if (proc == 0) seq_array = full_array; 

    // Последовательная версия.
    if (proc == 0) {
        auto t0 = std::chrono::high_resolution_clock::now();
        sequential_bubble_sort(seq_array);
        auto t1 = std::chrono::high_resolution_clock::now();
        seq_time = std::chrono::duration<double>(t1 - t0).count();
    }

    // Параллельная версия.
    if (num_procs > 1) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        MPI_Scatterv(
            proc == 0 ? full_array.data() : nullptr,
            sendcounts.data(), displs.data(), MPI_DOUBLE,
            local_array.data(), sendcounts[proc], MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        // Локальная пузырьковая сортировка.
        sequential_bubble_sort(local_array);

        std::vector<double> gathered;
        if (proc == 0) gathered.resize(N);
        MPI_Gatherv(
            local_array.data(), sendcounts[proc], MPI_DOUBLE,
            proc == 0 ? gathered.data() : nullptr,
            sendcounts.data(), displs.data(), MPI_DOUBLE,
            0, MPI_COMM_WORLD
        );

        // Слияние всех частей на процессе 0.
        if (proc == 0) {
            std::vector<double> merged;
            merged.reserve(N);
            for (int p = 0; p < num_procs; ++p) {
                merged.insert(
                    merged.end(),
                    gathered.begin() + displs[p],
                    gathered.begin() + displs[p] + sendcounts[p]
                );
            }
            std::inplace_merge(
                merged.begin(),
                merged.begin() + sendcounts[0],
                merged.end()
            );
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double par_time_iter = MPI_Wtime() - t_start;
        MPI_Reduce(&par_time_iter, &par_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    else if (proc == 0) {
        par_time = seq_time;
    }

    // Вывод результатов на процессе 0.
    if (proc == 0) {
        // Запись в файл.
        std::ofstream out("output_proc0_" + std::to_string(N) + ".out");
        out << std::fixed << std::setprecision(7)
            << "Array size: " << N << "\n"
            << "Number of processes: " << num_procs << "\n"
            << "Sequential version:\n"
            << "Time: " << seq_time << " seconds\n"
            << "Parallel version:\n"
            << "Time: " << par_time << " seconds\n";
        out.close();

        // Консольный вывод.
        std::cout << std::fixed << std::setprecision(7)
            << "Array size: " << N << "\n"
            << "Number of processes: " << num_procs << "\n"
            << "Sequential version:\n"
            << "Time: " << seq_time << " seconds\n"
            << "Parallel version:\n"
            << "Time: " << par_time << " seconds\n";
    }

    MPI_Finalize();
    return 0;
}
