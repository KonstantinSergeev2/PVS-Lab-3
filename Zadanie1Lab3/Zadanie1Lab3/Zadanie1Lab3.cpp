#include <iostream>
#include <vector>
#include <mpi.h>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <ctime>
#include <algorithm>

// Функция для последовательного вычисления суммы элементов массива.
double sequential_sum(const std::vector<double>& array) {
    double sum = 0.0;
    for (double val : array) {
        sum += val;
    }
    return sum;
}

int main(int argc, char* argv[]) {
    // Инициализация MPI.
    MPI_Init(&argc, &argv);

    int rank, size;
    // Получаем уникальный номер текущего процесса.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Получаем общее количество процессов, участвующих в вычислениях.
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = 0;  // Размер массива, который будет вычисляться.
    // Ввод размера массива только на процессе с рангом 0.
    if (rank == 0) {
        if (argc >= 2) {
            N = static_cast<size_t>(std::atoll(argv[1]));  // Конвертация строки в число.
            // Проверяем, что размер больше 100000.
            if (N <= 100000) {
                std::cerr << "Error: Array size must be greater than 100000.\n";
                // Завершаем выполнение всей программы с ошибкой.
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
        else {
            // Если аргумент не передан, запрашиваем ввод с клавиатуры.
            std::cout << "Enter array size (must be greater than 100000): ";
            std::cin >> N;
            if (N <= 100000) {
                std::cerr << "Error: Array size must be greater than 100000.\n";
                MPI_Abort(MPI_COMM_WORLD, 2);
            }
        }
    }

    // Рассылаем значение N всем процессам, чтобы все знали размер массива.
    MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

    // Инициализация полного массива только на процессе 0.
    std::vector<double> full_array;
    if (rank == 0) {
        full_array.resize(N);  // Выделяем память под N элементов.
        srand(static_cast<unsigned>(time(nullptr)));  // Инициализация генератора случайных чисел.
        // Заполняем массив случайными числами от 0 до 1.
        for (size_t i = 0; i < N; ++i) {
            full_array[i] = static_cast<double>(rand()) / RAND_MAX;
        }
    }

    double seq_time_sum = 0.0;
    double par_time_sum = 0.0;
    double seq_sum = 0.0;  // Итоговая сумма для последовательной версии.
    double par_sum = 0.0;  // Итоговая сумма для параллельной версии.

    const int repetitions = 100;  // Количество повторных запусков для замера времени.

    // Подготавливаем массивы для распределения данных по процессам.
    std::vector<int> sendcounts(size); // Сколько элементов отдавать каждому процессу.
    std::vector<int> displs(size);   // Смещения для каждого блока в исходном массиве.

    // Вычисляем базовое количество элементов на каждый процесс.
    size_t base = N / size;
    size_t remainder = N % size;
    size_t offset = 0;

    // Распределяем элементы между процессами с учётом остатка.
    for (int r = 0; r < size; ++r) {
        size_t count = base + (r < static_cast<int>(remainder) ? 1 : 0);
        sendcounts[r] = static_cast<int>(count);
        displs[r] = static_cast<int>(offset);
        offset += count;
    }

    // Локальный массив, в который попадут элементы для текущего процесса.
    std::vector<double> local_array(sendcounts[rank]);

    // Создаем копию массива для последовательных вычислений.
    // Это гарантирует одинаковые условия доступа к памяти.
    std::vector<double> seq_array;
    if (rank == 0) {
        seq_array = full_array;
    }

    for (int i = 0; i < repetitions; ++i) {
        // Последовательное вычисление суммы и замер времени.
        if (rank == 0) {
            sequential_sum(seq_array);
            auto t0 = std::chrono::high_resolution_clock::now(); // Запускаем таймер.
            double temp_sum = sequential_sum(seq_array); // Считаем сумму последовательно.
            auto t1 = std::chrono::high_resolution_clock::now(); // Останавливаем таймер.
            seq_time_sum += std::chrono::duration<double>(t1 - t0).count(); // Добавляем время к сумме.
            seq_sum = temp_sum;  // Запоминаем последний результат суммы для вывода.
        }

        MPI_Barrier(MPI_COMM_WORLD); // Синхронизация процессов перед параллельным замером.

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime(); // Запуск таймера MPI после синхронизации.

        // Распределяем части массива по процессам
        MPI_Scatterv(
            rank == 0 ? full_array.data() : nullptr, // Данные для рассылки (только на процессе 0).
            sendcounts.data(), // Кол-во элементов каждому процессу.
            displs.data(), // Смещения для каждого процесса.
            MPI_DOUBLE, // Тип данных.
            local_array.data(), // Буфер приёма для текущего процесса.
            sendcounts[rank], // Кол-во принимаемых элементов.
            MPI_DOUBLE,
            0, // Корневой процесс — отправитель.
            MPI_COMM_WORLD
        );

        double local_sum = 0.0;
        for (double val : local_array) {
            local_sum += val;
        }

        double temp_par_sum = 0.0;
        // Собираем все локальные суммы на процессе 0 (суммируем их).
        MPI_Reduce(&local_sum, &temp_par_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        // Синхронизируем все процессы перед окончанием замера.
        MPI_Barrier(MPI_COMM_WORLD);
        double par_time_iter = MPI_Wtime() - t_start;  // Локальное время для каждого процесса.

        // Находим максимальное время выполнения среди всех процессов.
        double max_par_time_iter;
        MPI_Reduce(
            &par_time_iter,
            &max_par_time_iter,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        // На процессе 0 накапливаем время и запоминаем сумму.
        if (rank == 0) {
            par_time_sum += max_par_time_iter;
            par_sum = temp_par_sum;
        }
    }

    // Выводим итоговые результаты. Среднее время и сумму для обеих версий.
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(7);
        std::cout
            << "\nArray size: " << N
            << "\nNumber of processes: " << size
            << "\n\nSequential version (average over " << repetitions << " runs):"
            << "\nSum: " << seq_sum
            << "\nAverage Time: " << (seq_time_sum / repetitions) << " seconds"
            << "\n\nParallel version (average over " << repetitions << " runs):"
            << "\nSum: " << par_sum
            << "\nAverage Time: " << (par_time_sum / repetitions) << " seconds\n";
    }

    // Завершаем работу MPI
    MPI_Finalize();
    return 0;
}