#include <algorithm>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <utility>
using namespace std;

vector<vector<char>> A;
vector<vector<bool>> visit;
set<pair<int, int>> empty_adj;
map<pair<int, int>, int> captured;

int rec(int r, int c) {
  visit[r][c] = true;
  int cnt = 1;
  for (auto [r2, c2] : {pair{r - 1, c}, {r, c - 1}, {r + 1, c}, {r, c + 1}}) {
    if (A[r2][c2] == '.') {
      empty_adj.insert({r2, c2});
    } else if (A[r2][c2] == 'W' && !visit[r2][c2]) {
      cnt += rec(r2, c2);
    }
  }
  return cnt;
}

int solve() {
  int R, C;
  cin >> R >> C;
  A.assign(R + 2, vector<char>(C + 2, 'B'));
  for (int i = 1; i <= R; i++) {
    for (int j = 1; j <= C; j++) {
      cin >> A[i][j];
    }
  }
  visit.assign(R + 2, vector<bool>(C + 2, false));
  captured.clear();
  int ans = 0;
  for (int i = 1; i <= R; i++) {
    for (int j = 1; j <= C; j++) {
      if (A[i][j] == 'W' && !visit[i][j]) {
        empty_adj.clear();
        int cnt = rec(i, j);
        if (empty_adj.size() == 1) {
          ans = max(ans, captured[*empty_adj.begin()] += cnt);
        }
      }
    }
  }
  return ans;
}

int main() {
  int T;
  cin >> T;
  for (int t = 1; t <= T; t++) {
    cout << "Case #" << t << ": " << solve() << endl;
  }
  return 0;
}