#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <ctime>

// Функция для последовательного вычисления суммы элементов массива
double sequential_sum(const std::vector<double>& array) {
    double sum = 0.0;
    for (size_t i = 0; i < array.size(); ++i) {
        sum += array[i]; // Суммирование элементов
    }
    return sum;
}

// Основная функция программы
int main(int argc, char* argv[]) {
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    // Получение ранга процесса и числа процессов
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ввод размера массива и желаемого числа процессов на процессе 0
    size_t N = 0;
    int desired_procs = 0;
    if (rank == 0) {
        // Запрос размера массива
        std::cout << "Array size: ";
        std::cin >> N;
        while (N <= 100000) {
            std::cout << "Array size must be > 100000. Try again: ";
            std::cin >> N;
        }

        // Запрос желаемого числа процессов
        std::cout << "Enter number of processes: ";
        std::cin >> desired_procs;
    }

    // Передача размера массива всем процессам
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Создание и заполнение массива на процессе 0
    std::vector<double> array;
    if (rank == 0) {
        array.resize(N);
        srand(static_cast<unsigned>(time(nullptr))); // Seed для случайных чисел
        for (size_t i = 0; i < N; ++i) {
            array[i] = static_cast<double>(rand()) / RAND_MAX; // Числа от 0 до 1
        }
    }

    // Последовательное вычисление суммы (на процессе 0)
    double seq_sum = 0.0;
    double seq_time = 0.0;
    if (rank == 0) {
        auto start = std::chrono::high_resolution_clock::now();
        seq_sum = sequential_sum(array);
        auto end = std::chrono::high_resolution_clock::now();
        seq_time = std::chrono::duration<double>(end - start).count();
    }

    // Вычисление локального размера массива для каждого процесса
    size_t local_n = N / size;
    size_t remainder = N % size;
    if (static_cast<size_t>(rank) < remainder) {
        ++local_n; // Учёт остатка
    }

    // Подготовка данных для распределения массива
    std::vector<int> sendcounts(size, 0);
    std::vector<int> displs(size, 0);
    size_t offset = 0;
    for (int r = 0; r < size; ++r) {
        size_t r_local_n = N / size;
        if (static_cast<size_t>(r) < remainder) ++r_local_n;
        sendcounts[r] = static_cast<int>(r_local_n);
        displs[r] = static_cast<int>(offset);
        offset += r_local_n;
    }

    // Локальный массив для каждого процесса
    std::vector<double> local_array(local_n);

    // Замер времени для параллельной версии
    double start_time = MPI_Wtime();

    // Распределение массива между процессами
    MPI_Scatterv(rank == 0 ? array.data() : nullptr, sendcounts.data(), displs.data(), MPI_DOUBLE,
        local_array.data(), local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Вычисление локальной суммы
    double local_sum = 0.0;
    for (size_t i = 0; i < local_n; ++i) {
        local_sum += local_array[i];
    }

    // Сбор общей суммы на процессе 0
    double par_sum = 0.0;
    MPI_Reduce(&local_sum, &par_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Остановка замера времени
    double par_time = MPI_Wtime() - start_time;

    // Вывод результатов на процессе 0
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Sequential time: " << seq_time << " seconds, sum: " << seq_sum << std::endl;
        std::cout << "Parallel time:   " << par_time << " seconds, sum: " << par_sum << std::endl;
    }

    // Завершение MPI
    MPI_Finalize();
    return 0;
}