import numpy as np


class GaussSeidel:
    """Class untuk menyelesaikan sistem Ax = b
    menggunakan metode iteratif Gauss-Seidel (konsep OOP)
    """

    def __init__(self, A, b, x0=None, tol=1e-6, max_iter=100):
        """Constructor: inisialisasi data awal objek"""
        self.A = np.array(A, dtype=float)   # matriks koefisien
        self.b = np.array(b, dtype=float)   # vektor konstanta
        self.n = len(b)                     # jumlah variabel

        # tebakan awal solusi
        self.x = np.zeros(self.n) if x0 is None else np.array(x0, dtype=float)

        self.tol = tol              # batas error (kriteria berhenti)
        self.max_iter = max_iter    # maksimum iterasi

        # penyimpanan proses iterasi
        self.history = []           # simpan hasil tiap iterasi
        self.errors = []            # simpan error tiap iterasi
        self.iterations = 0         # jumlah iterasi
        self.converged = False      # status konvergensi

    def cek_diagonal_dominan(self):
        """Cek apakah matriks diagonal dominan (syarat konvergensi)"""
        for i in range(self.n):
            diagonal = abs(self.A[i, i])                       # elemen diagonal
            jumlah_lain = np.sum(np.abs(self.A[i])) - diagonal # jumlah selain diagonal
            if diagonal < jumlah_lain:
                return False
        return True

    def solve(self):
        """Method utama menjalankan iterasi Gauss-Seidel"""
        
        # peringatan jika tidak memenuhi syarat
        if not self.cek_diagonal_dominan():
            print("Peringatan: matriks tidak diagonal dominan")

        x = self.x.copy()  # ambil nilai awal

        # proses iterasi
        for k in range(self.max_iter):
            x_old = x.copy()  # simpan nilai sebelumnya

            for i in range(self.n):
                # Gauss-Seidel: pakai nilai terbaru (kiri) dan lama (kanan)
                sum_kiri = np.dot(self.A[i, :i], x[:i])           # nilai baru
                sum_kanan = np.dot(self.A[i, i+1:], x_old[i+1:])  # nilai lama

                x[i] = (self.b[i] - sum_kiri - sum_kanan) / self.A[i, i]

            # hitung error (norm tak hingga)
            error = np.linalg.norm(x - x_old, ord=np.inf)

            # simpan proses
            self.history.append(x.copy())
            self.errors.append(error)

            # cek konvergensi
            if error < self.tol:
                self.converged = True
                self.iterations = k + 1
                self.x = x
                return self.x

        # jika tidak konvergen
        self.x = x
        self.iterations = self.max_iter
        return self.x

    def tampilkan_iterasi(self):
        """Menampilkan hasil setiap iterasi"""
        print("\n=== ITERASI GAUSS-SEIDEL ===")
        for i, nilai in enumerate(self.history, start=1):
            nilai_str = "  ".join([f"{v:.6f}" for v in nilai])
            print(f"Iterasi {i}: {nilai_str} | error = {self.errors[i-1]:.6e}")

    def hasil(self):
        """Menampilkan hasil akhir"""
        print("\nHASIL METODE GAUSS-SEIDEL")
        for i, val in enumerate(self.x, start=1):
            print(f"x{i} = {val:.8f}")
        print(f"Iterasi     : {self.iterations}")
        print(f"Error akhir : {self.errors[-1]:.8e}")
        print(f"Status      : {'Konvergen' if self.converged else 'Tidak konvergen'}")

    def residu(self):
        """Menghitung Ax - b"""
        return np.dot(self.A, self.x) - self.b