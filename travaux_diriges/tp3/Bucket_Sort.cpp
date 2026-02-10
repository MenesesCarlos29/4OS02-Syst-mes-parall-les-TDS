#include <mpi.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct TimingStats {
    double min_time = 0.0;
    double avg_time = 0.0;
    double max_time = 0.0;
};

void quicksort(std::vector<double>& data, int left, int right) {
    while (left < right) {
        int i = left;
        int j = right;
        double pivot = data[left + (right - left) / 2];

        while (i <= j) {
            while (data[i] < pivot) {
                ++i;
            }
            while (data[j] > pivot) {
                --j;
            }
            if (i <= j) {
                std::swap(data[i], data[j]);
                ++i;
                --j;
            }
        }

        if (j - left < right - i) {
            if (left < j) {
                quicksort(data, left, j);
            }
            left = i;
        } else {
            if (i < right) {
                quicksort(data, i, right);
            }
            right = j;
        }
    }
}

TimingStats reduce_timing_stats(double local_time, int rank, int size) {
    TimingStats stats;
    double sum_time = 0.0;

    MPI_Reduce(&local_time, &stats.min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &stats.max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_time, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        stats.avg_time = sum_time / static_cast<double>(size);
    }

    return stats;
}

int main(int argc, char** argv) {
    // ---------------------------------------------------------------------
    // Step 1: Initialization and parameter setup
    // ---------------------------------------------------------------------
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 100;  // Default global array size
    bool enable_debug = false;  // Enable with second argument: 1
    int quicksort_runs = 3;  // More than one run gives a stabler sequential baseline
    if (argc > 1) {
        N = std::atoi(argv[1]);
    }
    if (argc > 2) {
        enable_debug = (std::atoi(argv[2]) != 0);
    }
    if (argc > 3) {
        quicksort_runs = std::max(1, std::atoi(argv[3]));
    }

    if (rank == 0) {
        if (N < size) {
            std::cout << "Process 0: N (" << N << ") < size (" << size
                      << "), adjusting N to " << size << " to avoid empty local blocks.\n";
            N = size;
        }

        if (N % size != 0) {
            int adjusted = (N / size) * size;
            std::cout << "Process 0: N (" << N << ") is not divisible by size (" << size
                      << "), truncating to " << adjusted << ".\n";
            N = adjusted;
        }
    }

    // Broadcast adjusted N to all ranks
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = (size > 0) ? (N / size) : 0;

    enum PhaseId {
        PHASE_GENERATE = 0,
        PHASE_SCATTER,
        PHASE_LOCAL_SORT_1,
        PHASE_SAMPLE_ALLGATHER,
        PHASE_DEFINE_PIVOTS,
        PHASE_BUCKET_PREP,
        PHASE_COUNTS_ALLTOALL,
        PHASE_DATA_ALLTOALLV,
        PHASE_LOCAL_SORT_2,
        PHASE_GATHER_COUNTS,
        PHASE_GATHERV_DATA,
        PHASE_COUNT
    };

    std::array<double, PHASE_COUNT> phase_times{};

    auto time_phase = [&](int phase_id, const auto& phase_fn) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        phase_fn();
        MPI_Barrier(MPI_COMM_WORLD);
        phase_times[phase_id] = MPI_Wtime() - t0;
    };

    auto print_ordered = [&](const std::string& message) {
        for (int r = 0; r < size; ++r) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (r == rank) {
                std::cout << message << std::flush;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    };

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();

    // ---------------------------------------------------------------------
    // Step 2: Data generation and distribution (Scatter)
    // ---------------------------------------------------------------------
    std::vector<double> global_data;
    std::vector<double> quicksort_input_data;
    time_phase(PHASE_GENERATE, [&]() {
        if (rank == 0) {
            global_data.resize(N);
            std::mt19937 rng(2);
            std::uniform_real_distribution<double> dist(0.0, 1.0);
            for (int i = 0; i < N; ++i) {
                global_data[i] = dist(rng);
            }
            quicksort_input_data = global_data;
        }
    });

    std::vector<double> local_buffer(local_n);
    time_phase(PHASE_SCATTER, [&]() {
        MPI_Scatter(rank == 0 ? global_data.data() : nullptr,
                    local_n,
                    MPI_DOUBLE,
                    local_buffer.data(),
                    local_n,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
    });

    if (enable_debug) {
        print_ordered("Process " + std::to_string(rank) + ": received " +
                      std::to_string(local_n) + " elements.\n");
    }

    // ---------------------------------------------------------------------
    // Step 3: Local presort and sampling
    // ---------------------------------------------------------------------
    time_phase(PHASE_LOCAL_SORT_1, [&]() {
        std::sort(local_buffer.begin(), local_buffer.end());
    });

    // Option 2 (canonical regular sampling):
    // each process picks p-1 interior samples to approximate j/p quantiles.
    int samples_per_process = (size > 1) ? (size - 1) : 1;
    std::vector<double> local_samples(samples_per_process);
    std::vector<double> all_samples(samples_per_process * size);
    time_phase(PHASE_SAMPLE_ALLGATHER, [&]() {
        for (int i = 0; i < samples_per_process; ++i) {
            int quantile_id = i + 1;  // 1..p-1
            int idx = (quantile_id * local_n) / size;
            if (idx >= local_n) {
                idx = local_n - 1;
            }
            local_samples[i] = local_buffer[idx];
        }

        MPI_Allgather(local_samples.data(),
                      samples_per_process,
                      MPI_DOUBLE,
                      all_samples.data(),
                      samples_per_process,
                      MPI_DOUBLE,
                      MPI_COMM_WORLD);
    });

    // ---------------------------------------------------------------------
    // Step 4: Splitter (pivot) selection
    // ---------------------------------------------------------------------
    std::vector<double> pivots(size - 1);
    time_phase(PHASE_DEFINE_PIVOTS, [&]() {
        std::sort(all_samples.begin(), all_samples.end());
        for (int i = 1; i < size; ++i) {
            int pivot_sample_index = i * samples_per_process - 1;
            pivots[i - 1] = all_samples[pivot_sample_index];
        }
    });

    // ---------------------------------------------------------------------
    // Step 5: Exchange preparation (bucketing)
    // ---------------------------------------------------------------------
    std::vector<int> send_counts(size, 0);
    std::vector<int> send_displs(size, 0);

    time_phase(PHASE_BUCKET_PREP, [&]() {
        int prev_index = 0;
        for (int i = 0; i < size - 1; ++i) {
            int idx = static_cast<int>(
                std::upper_bound(local_buffer.begin(), local_buffer.end(), pivots[i]) -
                local_buffer.begin());
            send_counts[i] = idx - prev_index;
            prev_index = idx;
        }
        send_counts[size - 1] = local_n - prev_index;

        for (int i = 1; i < size; ++i) {
            send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
        }
    });

    std::vector<int> recv_counts(size, 0);
    time_phase(PHASE_COUNTS_ALLTOALL, [&]() {
        MPI_Alltoall(
            send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    });

    std::vector<int> recv_displs(size, 0);
    for (int i = 1; i < size; ++i) {
        recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
    }

    int total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);

    if (enable_debug) {
        print_ordered("Process " + std::to_string(rank) + ": sending " +
                      std::to_string(local_n) + " elements in total, receiving " +
                      std::to_string(total_recv) + ".\n");
    }

    // ---------------------------------------------------------------------
    // Step 6: Data exchange (Alltoallv)
    // ---------------------------------------------------------------------
    std::vector<double> sorted_bucket(total_recv);
    time_phase(PHASE_DATA_ALLTOALLV, [&]() {
        MPI_Alltoallv(local_buffer.data(),
                      send_counts.data(),
                      send_displs.data(),
                      MPI_DOUBLE,
                      sorted_bucket.data(),
                      recv_counts.data(),
                      recv_displs.data(),
                      MPI_DOUBLE,
                      MPI_COMM_WORLD);
    });

    // ---------------------------------------------------------------------
    // Step 7: Final local sort
    // ---------------------------------------------------------------------
    time_phase(PHASE_LOCAL_SORT_2, [&]() {
        std::sort(sorted_bucket.begin(), sorted_bucket.end());
    });

    // ---------------------------------------------------------------------
    // Step 8: Final collection (Gatherv)
    // ---------------------------------------------------------------------
    int local_sorted_n = static_cast<int>(sorted_bucket.size());
    std::vector<int> final_counts;
    std::vector<int> final_displs;
    std::vector<double> global_sorted_data;

    if (rank == 0) {
        final_counts.resize(size, 0);
    }

    time_phase(PHASE_GATHER_COUNTS, [&]() {
        MPI_Gather(&local_sorted_n,
                   1,
                   MPI_INT,
                   rank == 0 ? final_counts.data() : nullptr,
                   1,
                   MPI_INT,
                   0,
                   MPI_COMM_WORLD);
    });

    if (rank == 0) {
        final_displs.resize(size, 0);
        for (int i = 1; i < size; ++i) {
            final_displs[i] = final_displs[i - 1] + final_counts[i - 1];
        }
        int total_final = std::accumulate(final_counts.begin(), final_counts.end(), 0);
        global_sorted_data.resize(total_final);
    }

    time_phase(PHASE_GATHERV_DATA, [&]() {
        MPI_Gatherv(sorted_bucket.data(),
                    local_sorted_n,
                    MPI_DOUBLE,
                    rank == 0 ? global_sorted_data.data() : nullptr,
                    rank == 0 ? final_counts.data() : nullptr,
                    rank == 0 ? final_displs.data() : nullptr,
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
    });

    MPI_Barrier(MPI_COMM_WORLD);
    double local_total_time = MPI_Wtime() - total_start;

    // Final validation on rank 0
    if (rank == 0) {
        bool ok = std::is_sorted(global_sorted_data.begin(), global_sorted_data.end());
        std::cout << (ok ? "Success: the global array is sorted.\n"
                         : "Error: the global array is NOT sorted.\n");
    }

    // Comparison with sequential quicksort on rank 0
    double quicksort_time_min = 0.0;
    double quicksort_time_avg = 0.0;
    bool quicksort_ok = true;
    if (rank == 0) {
        std::vector<double> quicksort_times(quicksort_runs, 0.0);
        for (int run = 0; run < quicksort_runs; ++run) {
            std::vector<double> run_data = quicksort_input_data;
            double t0 = MPI_Wtime();
            if (!run_data.empty()) {
                quicksort(run_data, 0, static_cast<int>(run_data.size()) - 1);
            }
            quicksort_times[run] = MPI_Wtime() - t0;
            quicksort_ok = quicksort_ok && std::is_sorted(run_data.begin(), run_data.end());
        }
        quicksort_time_min = *std::min_element(quicksort_times.begin(), quicksort_times.end());
        quicksort_time_avg =
            std::accumulate(quicksort_times.begin(), quicksort_times.end(), 0.0) /
            static_cast<double>(quicksort_runs);
    }

    // ---------------------------------------------------------------------
    // Performance and efficiency metrics
    // ---------------------------------------------------------------------
    std::array<TimingStats, PHASE_COUNT> phase_stats;
    for (int i = 0; i < PHASE_COUNT; ++i) {
        phase_stats[i] = reduce_timing_stats(phase_times[i], rank, size);
    }
    TimingStats total_stats = reduce_timing_stats(local_total_time, rank, size);

    long long local_sent_elements = static_cast<long long>(local_n);
    long long local_recv_elements = static_cast<long long>(total_recv);
    long long local_stay_elements = static_cast<long long>(send_counts[rank]);
    long long global_sent_elements = 0;
    long long global_recv_elements = 0;
    long long global_stay_elements = 0;
    MPI_Reduce(
        &local_sent_elements, &global_sent_elements, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(
        &local_recv_elements, &global_recv_elements, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(
        &local_stay_elements, &global_stay_elements, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const std::array<const char*, PHASE_COUNT> phase_names = {
            "Step2_Generation",
            "Step2_Scatter",
            "Step3_LocalSort",
            "Step3_Sampling_Allgather",
            "Step4_Splitters",
            "Step5_BucketPartitioning",
            "Step5_Alltoall_Counts",
            "Step6_Alltoallv_Data",
            "Step7_FinalLocalSort",
            "Step8_Gather_Counts",
            "Step8_Gatherv_Data"};

        std::cout << "\n===== Performance Report =====\n";
        std::cout << "N = " << N << ", MPI processes = " << size << "\n";
        std::cout << std::left << std::setw(30) << "Phase"
                  << std::right << std::setw(14) << "min(s)"
                  << std::setw(14) << "avg(s)"
                  << std::setw(14) << "max(s)"
                  << std::setw(16) << "eff(avg/max)\n";

        for (int i = 0; i < PHASE_COUNT; ++i) {
            double phase_efficiency = (phase_stats[i].max_time > 0.0)
                                          ? (phase_stats[i].avg_time / phase_stats[i].max_time)
                                          : 0.0;
            std::cout << std::left << std::setw(30) << phase_names[i]
                      << std::right << std::setw(14) << std::fixed << std::setprecision(6)
                      << phase_stats[i].min_time << std::setw(14) << phase_stats[i].avg_time
                      << std::setw(14) << phase_stats[i].max_time << std::setw(16)
                      << phase_efficiency << "\n";
        }

        double communication_max =
            phase_stats[PHASE_SCATTER].max_time + phase_stats[PHASE_SAMPLE_ALLGATHER].max_time +
            phase_stats[PHASE_COUNTS_ALLTOALL].max_time + phase_stats[PHASE_DATA_ALLTOALLV].max_time +
            phase_stats[PHASE_GATHER_COUNTS].max_time + phase_stats[PHASE_GATHERV_DATA].max_time;
        double sorting_max =
            phase_stats[PHASE_LOCAL_SORT_1].max_time + phase_stats[PHASE_LOCAL_SORT_2].max_time;
        double communication_ratio = (total_stats.max_time > 0.0)
                                         ? (100.0 * communication_max / total_stats.max_time)
                                         : 0.0;

        int max_bucket = *std::max_element(final_counts.begin(), final_counts.end());
        int min_bucket = *std::min_element(final_counts.begin(), final_counts.end());
        double avg_bucket = static_cast<double>(N) / static_cast<double>(size);
        double load_balance_efficiency =
            (max_bucket > 0) ? (avg_bucket / static_cast<double>(max_bucket)) : 0.0;
        double bucket_variance = 0.0;
        for (int count : final_counts) {
            double delta = static_cast<double>(count) - avg_bucket;
            bucket_variance += delta * delta;
        }
        bucket_variance /= static_cast<double>(size);
        double bucket_stddev = std::sqrt(bucket_variance);
        double bucket_cv = (avg_bucket > 0.0) ? (bucket_stddev / avg_bucket) : 0.0;
        double bucket_imbalance_ratio =
            (min_bucket > 0) ? (static_cast<double>(max_bucket) / static_cast<double>(min_bucket))
                             : 0.0;
        long long moved_elements = static_cast<long long>(N) - global_stay_elements;
        long long moved_bytes = moved_elements * static_cast<long long>(sizeof(double));
        double moved_ratio =
            (N > 0) ? (100.0 * static_cast<double>(moved_elements) / static_cast<double>(N)) : 0.0;
        double parallel_throughput_melems =
            (total_stats.max_time > 0.0) ? (static_cast<double>(N) / 1e6) / total_stats.max_time : 0.0;
        double quicksort_throughput_melems =
            (quicksort_time_min > 0.0) ? (static_cast<double>(N) / 1e6) / quicksort_time_min : 0.0;

        std::cout << "\nTotal time (max across ranks): " << std::fixed << std::setprecision(6)
                  << total_stats.max_time << " s\n";
        std::cout << "Communication time (approx, sum of max): " << communication_max << " s ("
                  << communication_ratio << "% of total max)\n";
        std::cout << "Local sorting time (sum of max): " << sorting_max << " s\n";
        std::cout << "Globally sent data: "
                  << (global_sent_elements * static_cast<long long>(sizeof(double))) << " bytes\n";
        std::cout << "Globally received data: "
                  << (global_recv_elements * static_cast<long long>(sizeof(double))) << " bytes\n";
        std::cout << "Data moved across ranks (Alltoallv, approx): " << moved_bytes
                  << " bytes (" << moved_ratio << "% of N)\n";
        std::cout << "Final bucket balance (min/avg/max): " << min_bucket << " / " << avg_bucket
                  << " / " << max_bucket << "\n";
        std::cout << "Balance efficiency (avg/max): " << load_balance_efficiency << "\n";
        std::cout << "Bucket stddev: " << bucket_stddev << " | bucket CV: " << bucket_cv
                  << " | max/min: " << bucket_imbalance_ratio << "\n";
        std::cout << "Parallel throughput (end-to-end): " << parallel_throughput_melems
                  << " Melem/s\n";

        std::cout << "\n===== Quicksort Comparison =====\n";
        std::cout << "Quicksort time (rank 0, sequential, min/" << quicksort_runs
                  << "): " << std::fixed << std::setprecision(6) << quicksort_time_min << " s\n";
        std::cout << "Quicksort time (rank 0, sequential, avg/" << quicksort_runs
                  << "): " << quicksort_time_avg << " s\n";
        if (!quicksort_ok) {
            std::cout << "Quicksort: ERROR (output not sorted)\n";
        }
        double speedup =
            (total_stats.max_time > 0.0) ? (quicksort_time_min / total_stats.max_time) : 0.0;
        double efficiency = (size > 0) ? (speedup / static_cast<double>(size)) : 0.0;
        std::cout << "Throughput quicksort (min): " << quicksort_throughput_melems << " Melem/s\n";
        std::cout << "Speedup (quicksort / parallel): " << speedup << "\n";
        std::cout << "Parallel efficiency (speedup / p): " << efficiency << "\n";
    }

    // Cleanup
    MPI_Finalize();
    return 0;
}
