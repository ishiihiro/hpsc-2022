#include <cmath>
#include <vector>

using namespace std;

int main(int argc, char **argv) {
  const int nx = 41;
  const int ny = 41;
  const int nt = 500;
  const int nit = 50;
  const double dx = 2.0 / (nx - 1);
  const double dy = 2.0 / (ny - 1);
  const double dt = 0.01;
  const int rho = 1;
  const double nu = 0.02;

  vector<vector<double>> u(ny, vector<double>(nx));
  vector<vector<double>> v(ny, vector<double>(nx));
  vector<vector<double>> p(ny, vector<double>(nx));
  vector<vector<double>> b(ny, vector<double>(nx));

  for(int n=0; n<nt; n++){
    #pragma omp parallel for collapse(2)
    for(int j=1; j<ny-1; j++)
      for(int i=1; i<nx-1; i++){
        b[j][i] = rho * (1 / dt * 
                ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) - 
                pow((u[j][i+1] - u[j][i-1]) / (2 * dx), 2) - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) * 
                (v[j][i+1] - v[j][i-1]) / (2 * dx)) - pow((v[j+1][i] - v[j-1][i]) / (2 * dy), 2));
      }
    for(int it=0; it<nit; it++){
      vector<vector<double>> pn(ny, vector<double>(nx));
      copy(&p[0][0], &p[0][0]+ny*nx, &pn[0][0]);
      #pragma omp parallel for collapse(2)
      for(int j=1; j<ny-1; j++)
        for(int i=1; i<nx-1; i++){
          p[j][i] = (pow(dy, 2) * (pn[j][i+1] + pn[j][i-1]) +
                     pow(dx, 2) * (pn[j+1][i] + pn[j-1][i]) -
                     b[j][i] * pow(dx, 2) * pow(dy, 2)) / (2 * (pow(dx, 2) + pow(dy,2)));
        }
      #pragma omp parallel for
      for(int j=0; j<ny; j++) p[j][nx-1] = p[j][nx-2];
      copy(p[1].begin(), p[1].end(), p[0].begin());
      #pragma omp parallel for
      for(int j=0; j<ny; j++) p[j][0] = p[j][1];
      fill(p[ny-1].begin(), p[ny-1].end(), 0);
    }
    vector<vector<double>> un(ny, vector<double>(nx));
    vector<vector<double>> vn(ny, vector<double>(nx));
    copy(&u[0][0], &u[0][0]+ny*nx, &un[0][0]);
    copy(&v[0][0], &v[0][0]+ny*nx, &vn[0][0]);
    #pragma omp parallel for collapse(2)
    for(int j=1; j<ny-1; j++)
      for(int i=1; i<nx-1; i++){
        u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i - 1])
                           - un[j][i] * dt / dy * (un[j][i] - un[j - 1][i])
                           - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                           + nu * dt / pow(dx, 2) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                           + nu * dt / pow(dy, 2) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
        v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i - 1])
                           - vn[j][i] * dt / dy * (vn[j][i] - vn[j - 1][i])
                           - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                           + nu * dt / pow(dx, 2) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                           + nu * dt / pow(dy, 2) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
      }
    fill(u[0].begin(), u[0].end(), 0);
    #pragma omp parallel for
    for(int j=0; j<ny; j++){
      u[j][0] = 0;
      u[j][nx-1] = 0;
    }
    fill(u[ny-1].begin(), u[ny-1].end(), 1);
    fill(v[0].begin(), v[0].end(), 0);
    fill(v[ny-1].begin(), v[ny-1].end(), 0);
    #pragma omp parallel for
    for(int j=0; j<ny; j++){
      v[j][0] = 0;
      v[j][nx-1] = 0;
    }
  }
}
