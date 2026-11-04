// Native XGBoost baseline: times XGBoosterPredictFromDense through the C API,
// with no Python in the loop.
//
// Build (the shared library ships inside the xgboost Python wheel):
//   XGB=$(python3 -c "import xgboost,os;print(os.path.dirname(xgboost.__file__))")/lib
//   gcc -O2 -o /tmp/xgb_capi bench/xgb_capi_bench.c -L$XGB -lxgboost -Wl,-rpath,$XGB
// Run:
//   /tmp/xgb_capi bench/data/medium.json bench/data/input.csv [threads] [repeat]
//
// Prints µs per row for a 10k-row batch and for one row at a time, and writes
// the batch predictions' float32 bit patterns to stdout's last line checksum.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void *BoosterHandle;
typedef uint64_t bst_ulong;
int XGBoosterCreate(const void *dmats[], bst_ulong len, BoosterHandle *out);
int XGBoosterLoadModel(BoosterHandle handle, const char *fname);
int XGBoosterSetParam(BoosterHandle handle, const char *name, const char *value);
int XGBoosterPredictFromDense(BoosterHandle handle, const char *values,
                              const char *config, void *m,
                              bst_ulong const **out_shape, bst_ulong *out_dim,
                              const float **out_result);
const char *XGBGetLastError(void);

#define CHECK(x)                                                    \
  do {                                                              \
    if ((x) != 0) {                                                 \
      fprintf(stderr, "%s: %s\n", #x, XGBGetLastError());           \
      exit(1);                                                      \
    }                                                               \
  } while (0)

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

static float *read_csv(const char *path, int *rows, int *cols) {
  FILE *f = fopen(path, "r");
  if (!f) { perror(path); exit(1); }
  size_t cap = 1 << 20, n = 0;
  float *v = malloc(cap * sizeof(float));
  char line[1 << 16];
  int r = 0, c = 0;
  while (fgets(line, sizeof line, f)) {
    int k = 0;
    for (char *tok = strtok(line, ",\n"); tok; tok = strtok(NULL, ",\n")) {
      if (n == cap) v = realloc(v, (cap *= 2) * sizeof(float));
      v[n++] = (strncmp(tok, "NaN", 3) == 0 || strncmp(tok, "nan", 3) == 0)
                   ? NAN : strtof(tok, NULL);
      k++;
    }
    if (k == 0) continue;
    c = k;
    r++;
  }
  fclose(f);
  *rows = r;
  *cols = c;
  return v;
}

static void array_interface(char *buf, size_t len, const float *p, int rows,
                            int cols) {
  snprintf(buf, len,
           "{\"data\":[%llu,true],\"shape\":[%d,%d],\"typestr\":\"<f4\","
           "\"version\":3}",
           (unsigned long long)(uintptr_t)p, rows, cols);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s model.json input.csv [threads] [repeat]\n", argv[0]);
    return 2;
  }
  const char *threads = argc > 3 ? argv[3] : "1";
  int repeat = argc > 4 ? atoi(argv[4]) : 5;
  int rows, cols;
  float *x = read_csv(argv[2], &rows, &cols);

  BoosterHandle b;
  CHECK(XGBoosterCreate(NULL, 0, &b));
  CHECK(XGBoosterLoadModel(b, argv[1]));
  CHECK(XGBoosterSetParam(b, "nthread", threads));
  const char *cfg =
      "{\"type\":0,\"training\":false,\"iteration_begin\":0,"
      "\"iteration_end\":0,\"strict_shape\":false,\"missing\":NaN}";
  char ai[256];
  const bst_ulong *shape;
  bst_ulong dim;
  const float *out;

  // Batch: whole matrix per call; best of `repeat`.
  array_interface(ai, sizeof ai, x, rows, cols);
  CHECK(XGBoosterPredictFromDense(b, ai, cfg, NULL, &shape, &dim, &out));
  double best = 1e30;
  for (int i = 0; i < repeat; i++) {
    double t0 = now();
    CHECK(XGBoosterPredictFromDense(b, ai, cfg, NULL, &shape, &dim, &out));
    double t = now() - t0;
    if (t < best) best = t;
  }
  uint64_t sum = 0;
  for (int i = 0; i < rows; i++) {
    uint32_t u;
    memcpy(&u, &out[i], 4);
    sum = sum * 1000003u + u;
  }
  printf("batch  %d rows: %.3f us/row\n", rows, best * 1e6 / rows);

  // One row per call, as a request handler would do it.
  int single = rows < 2000 ? rows : 2000;
  double t0 = now();
  for (int r = 0; r < single; r++) {
    array_interface(ai, sizeof ai, x + (size_t)r * cols, 1, cols);
    CHECK(XGBoosterPredictFromDense(b, ai, cfg, NULL, &shape, &dim, &out));
  }
  printf("single %d rows: %.3f us/row\n", single, (now() - t0) * 1e6 / single);
  printf("checksum %016llx\n", (unsigned long long)sum);
  return 0;
}
