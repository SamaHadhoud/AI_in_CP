#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

using int64 = long long;

int N, M;
vector<vector<int>> A;
vector<vector<int64>> dpl, dpr;
vector<vector<vector<int64>>> dp;

int64 rec(int dir, int x, int y) {
  if (dp[dir][x][y] != -1LL) {
    return dp[dir][x][y];
  }
  int64 res = dpr[0][0];
  if (dir) {
    int64 extra_exit = (y + 2 < M) ? dpr[x][y + 2] : 0;
    int64 max_cost = extra_exit;
    for (int ty = y; ty >= 0; --ty) {
      int64 best_entrance = 0LL;
      if (x) {
        best_entrance = max(best_entrance, dpl[x - 1][ty]);
        if (ty != y) {
          best_entrance = max(best_entrance, dpl[x - 1][ty + 1]);
        }
      }
      int64 best_exit = extra_exit;
      if (x + 1 != N && ty + 1 != M) {
        best_exit = max(best_exit, dpr[x + 1][ty + 1]);
      }
      max_cost = max(max_cost, best_entrance + best_exit);
      if (ty) {
        best_entrance = max(best_entrance, dpl[x][ty - 1]);
      }
      int64 local_max_cost = max(max_cost, best_entrance + best_exit);
      if (x + 1 != N || !ty) {
        int64 cur = (x + 1 == N) ? 0LL : rec(dir ^ 1, x + 1, ty);
        cur = max(cur, local_max_cost);
        res = min(res, cur);
      }
    }
  } else {
    int64 extra_entrance = (x - 2 >= 0) ? dpl[x - 2][y] : 0;
    int64 max_cost = extra_entrance;
    for (int tx = x; tx < N; ++tx) {
      int64 best_exit = 0LL;
      if (y + 1 != M) {
        best_exit = max(best_exit, dpr[tx][y + 1]);
        if (tx != x) {
          best_exit = max(best_exit, dpr[tx - 1][y + 1]);
        }
      }
      int64 best_entrance = extra_entrance;
      if (y && tx) {
        best_entrance = max(best_entrance, dpl[tx - 1][y - 1]);
      }
      max_cost = max(max_cost, best_entrance + best_exit);
      if (tx + 1 != N) {
        best_exit = max(best_exit, dpr[tx + 1][y]);
      }
      int64 local_max_cost = max(max_cost, best_entrance + best_exit);
      if (y || tx + 1 == N) {
        int64 cur = y ? rec(dir ^ 1, tx, y - 1) : 0LL;
        cur = max(cur, local_max_cost);
        res = min(res, cur);
      }
    }
  }
  return dp[dir][x][y] = res;
}

int64 solve() {
  cin >> N >> M;
  A.assign(N, vector<int>(M));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      cin >> A[i][j];
    }
  }
  dpl.assign(N, vector<int64>(M));
  dpr.assign(N, vector<int64>(M));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      dpl[i][j] = A[i][j];
      if (i) {
        dpl[i][j] = max(dpl[i][j], A[i][j] + dpl[i - 1][j]);
      }
      if (j) {
        dpl[i][j] = max(dpl[i][j], A[i][j] + dpl[i][j - 1]);
      }
    }
  }
  for (int i = N - 1; i >= 0; i--) {
    for (int j = M - 1; j >= 0; j--) {
      dpr[i][j] = A[i][j];
      if (i + 1 != N) {
        dpr[i][j] = max(dpr[i][j], A[i][j] + dpr[i + 1][j]);
      }
      if (j + 1 != M) {
        dpr[i][j] = max(dpr[i][j], A[i][j] + dpr[i][j + 1]);
      }
    }
  }
  dp.assign(2, vector<vector<int64>>(N, vector<int64>(M, -1LL)));
  return min(rec(0, 0, M - 1), rec(1, 0, M - 1));
}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}
