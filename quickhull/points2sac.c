// Convert a data file produces by gen_points.c into the data format
// expected by SAC.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/stat.h>

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: %s INFILE OUTFILE\n", argv[1]);
    exit(1);
  }

  FILE* fin = fopen(argv[1], "r");
  assert(fin != NULL);

  FILE* fout = fopen(argv[2], "w");
  assert(fout != NULL);

  struct stat statbuf;
  fstat(fileno(fin), &statbuf);

  int num_points = statbuf.st_size / (sizeof(int)*2);

  fprintf(fout, "[1,%d:\n", (int)num_points);
  for(int i = 0; i < num_points; i++) {
    int point[2];
    assert(fread(point, sizeof(int), 2, fin) == 2);
    fprintf(fout, "%d %d\n", point[0], point[1]);
  }
  fprintf(fout, "]\n");
  fclose(fout);
}
