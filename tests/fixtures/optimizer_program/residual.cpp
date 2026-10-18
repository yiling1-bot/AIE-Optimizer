using I16 = short;
using I32 = int;
void residual(const I16 (&a)[2][3], const I16 (&b)[2][3],
              I32 (& __restrict y)[2][3], I32 (& __restrict z)[2][3]) {
  for (int i=0;i<2;++i) for (int j=0;j<3;++j) {
    y[i][j]=(int)a[i][j]*(int)b[i][j]+(int)a[i][j];
  }
  for (int i=0;i<2;++i) for (int j=0;j<3;++j) {
    z[i][j]=(int)a[i][j]+(int)b[i][j];
  }
}
