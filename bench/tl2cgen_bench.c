// Times a TL2cgen-generated predictor on bench/data/input.csv (10000 rows,
// 32 features), one row at a time on one thread, and counts predictions
// that match the expected file bit for bit (as float32).
// Built by tl2cgen_bench.sh.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "header.h"

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  if (argc < 4) return 2;
  int n = 10000, f = 32;
  float *X = malloc(sizeof(float) * n * f), *E = malloc(sizeof(float) * n);
  FILE *fp = fopen(argv[1], "r");
  char buf[1 << 16];
  int r = 0;
  while (r < n && fgets(buf, sizeof buf, fp)) {
    char *s = buf;
    for (int j = 0; j < f; j++) {
      X[r * f + j] = strtof(s, &s);
      s++;
    }
    r++;
  }
  fclose(fp);
  fp = fopen(argv[2], "r");
  for (int i = 0; i < n; i++)
    if (fscanf(fp, "%f", &E[i]) != 1) return 1;
  fclose(fp);
  union Entry *row = malloc(sizeof(union Entry) * f);
  float *out = malloc(sizeof(float) * n);
  int reps = atoi(argv[3]);
  double best = 1e9;
  for (int k = 0; k < reps; k++) {
    double t0 = now();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < f; j++) {
        float v = X[i * f + j];
        if (isnan(v)) row[j].missing = -1;
        else row[j].fvalue = v;
      }
      float res = 0;
      predict(row, 0, &res);
      out[i] = res;
    }
    double t = (now() - t0) / n * 1e6;
    if (t < best) best = t;
  }
  int eq = 0;
  for (int i = 0; i < n; i++)
    if (memcmp(&out[i], &E[i], 4) == 0) eq++;
  printf("batch %.3f us/row, exact %d/%d\n", best, eq, n);
  return 0;
}
