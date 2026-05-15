import numpy as np


class Jacobi:
    """Class untuk menyelesaikan sistem persamaan linear Ax = b
    menggunakan metode iteratif Jacobi (konsep OOP: enkapsulasi data + metode)
    """

    def __init__(self, A, b, x0=None, tol=1e-6, max_iter=100):
        """Constructor: otomatis dipanggil saat objek dibuat"""
        self.A = np.array(A, dtype=float)   # matriks koefisien
        self.b = np.array(b, dtype=float)   # vektor hasil
        self.n = len(b)                     # jumlah variabel

        # inisialisasi tebakan awal
        self.x = np.zeros(self.n) if x0 is None else np.array(x0, dtype=float)

        self.tol = tol              # toleransi error (kriteria berhenti)
        self.max_iter = max_iter    # batas iterasi maksimum

        # atribut untuk menyimpan proses (state object)
        self.history = []           # menyimpan nilai tiap iterasi
        self.errors = []            # menyimpan error tiap iterasi
        self.iterations = 0         # jumlah iterasi yang dilakukan
        self.converged = False      # status konvergensi

    def cek_diagonal_dominan(self):
        """Method untuk mengecek apakah matriks diagonal dominan
        (syarat agar metode Jacobi cenderung konvergen)
        """
        for i in range(self.n):
            diagonal = abs(self.A[i, i])                      # nilai diagonal
            jumlah_lain = np.sum(np.abs(self.A[i])) - diagonal  # jumlah elemen lain
            if diagonal < jumlah_lain:
                return False
        return True

    def solve(self):
        """Method utama untuk menjalankan iterasi Jacobi"""
        
        # cek kondisi awal
        if not self.cek_diagonal_dominan():
            print("Peringatan: matriks tidak diagonal dominan")

        x_old = self.x.copy()  # simpan nilai sebelumnya

        # loop iterasi
        for k in range(self.max_iter):
            x_new = np.zeros(self.n)  # nilai baru

            # rumus Jacobi
            for i in range(self.n):
                sigma = np.dot(self.A[i], x_old) - self.A[i, i] * x_old[i]
                x_new[i] = (self.b[i] - sigma) / self.A[i, i]

            # hitung error (norm tak hingga)
            error = np.linalg.norm(x_new - x_old, ord=np.inf)

            # simpan history
            self.history.append(x_new.copy())
            self.errors.append(error)

            # cek konvergensi
            if error < self.tol:
                self.converged = True
                self.iterations = k + 1
                self.x = x_new
                return self.x

            x_old = x_new.copy()  # update nilai lama

        # jika tidak konvergen
        self.x = x_old
        self.iterations = self.max_iter
        return self.x

    def tampilkan_iterasi(self):
        """Menampilkan semua iterasi (output/debugging)"""
        print("\n=== ITERASI JACOBI ===")
        for i, nilai in enumerate(self.history, start=1):
            nilai_str = "  ".join([f"{v:.6f}" for v in nilai])
            print(f"Iterasi {i}: {nilai_str} | error = {self.errors[i-1]:.6e}")

    def hasil(self):
        """Menampilkan hasil akhir"""
        print("\nHASIL METODE JACOBI")
        for i, val in enumerate(self.x, start=1):
            print(f"x{i} = {val:.8f}")
        print(f"Iterasi     : {self.iterations}")
        print(f"Error akhir : {self.errors[-1]:.8e}")
        print(f"Status      : {'Konvergen' if self.converged else 'Tidak konvergen'}")

    def residu(self):
        """Menghitung residu Ax - b (cek akurasi solusi)"""
        return np.dot(self.A, self.x) - self.b