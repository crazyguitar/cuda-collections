#include <errno.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <string.h>
#include <unistd.h>

#include <cstdio>

#define CHECK(exp)                                                                            \
  do {                                                                                        \
    auto rc = (exp);                                                                          \
    if (rc < 0) {                                                                             \
      fprintf(stderr, "[%s:%d]" #exp "fail. err: %s\n", __FILE__, __LINE__, strerror(errno)); \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)

int main(int argc, char* argv[]) {
  constexpr size_t SIZE = 256;
  char hostname[SIZE] = {0};
  CHECK(gethostname(hostname, SIZE));
  nvshmem_init();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("[%s] mype: %d, npes: %d, mype_node: %d\n", hostname, mype, npes, mype_node);
  nvshmem_finalize();
}
