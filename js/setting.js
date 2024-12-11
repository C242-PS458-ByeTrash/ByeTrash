function previewProfilePic() {
    const fileInput = document.getElementById('profile-pic');
    const profileImg = document.getElementById('profile-img');
    const changePhotoBtn = document.getElementById('change-photo-btn');
    const file = fileInput.files[0];

    if (file) {
        const reader = new FileReader();
        reader.onload = function (e) {
            profileImg.src = e.target.result;
            profileImg.style.display = 'block';
            fileInput.style.display = 'none'; // Sembunyikan input file
            changePhotoBtn.style.display = 'inline-block'; // Tampilkan tombol Ubah Foto
        }
        reader.readAsDataURL(file);
    }
}

function changePhoto() {
    const fileInput = document.getElementById('profile-pic');
    const changePhotoBtn = document.getElementById('change-photo-btn');

    fileInput.style.display = 'block'; // Tampilkan input file
    changePhotoBtn.style.display = 'none'; // Sembunyikan tombol Ubah Foto
    fileInput.value = ''; // Reset nilai input file
}

// Add Firebase initialization and data fetching
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";
import { getFirestore, getDoc, doc } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-firestore.js";

const firebaseConfig = {
    apiKey: "AIzaSyDboHC-1yaGw7oVlg9BW555w6X0s6vLMtI",
    authDomain: "byetrash-442808.firebaseapp.com",
    projectId: "byetrash-442808",
    storageBucket: "byetrash-442808.firebasestorage.app",
    messagingSenderId: "913316825970",
    appId: "1:913316825970:web:95bed65a1fd02ba8cc2beb",
    measurementId: "G-29P29DH4YY"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth();
const db = getFirestore();

onAuthStateChanged(auth, (user) => {
    if (user) {
        const docRef = doc(db, "users", user.uid);
        getDoc(docRef).then((docSnap) => {
            if (docSnap.exists()) {
                const userData = docSnap.data();
                document.getElementById('name').innerText = `${userData.firstName} ${userData.lastName}`;
                document.getElementById('email').value = userData.email; // Populate email field
            }
        }).catch((error) => {
            console.error("Error getting document:", error);
        });
    } else {
        console.log("No user is signed in.");
    }
});

import { setDoc } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-firestore.js";

document.querySelector('form').addEventListener('submit', async (e) => {
    e.preventDefault(); // Mencegah form dari reload halaman

    const user = auth.currentUser;
    if (!user) {
        alert("Anda belum login!");
        return;
    }

    // Ambil data dari form
    const phone = document.getElementById('phone').value;
    const email = document.getElementById('email').value;
    const address = document.getElementById('address').value;
    const birthPlace = document.getElementById('birth-place').value;
    const birthDate = document.getElementById('birth-date').value;
    const age = document.getElementById('age').value;

    // Data yang akan disimpan
    const userData = {
        phone: phone,
        email: email,
        address: address,
        birthPlace: birthPlace,
        birthDate: birthDate,
        age: parseInt(age),
    };

    try {
        // Simpan ke Firestore
        const docRef = doc(db, "users", user.uid); // Gunakan UID pengguna saat ini sebagai ID dokumen
        await setDoc(docRef, userData, { merge: true }); // merge: true untuk memperbarui tanpa menimpa data lama
        alert("Data berhasil disimpan!");
    } catch (error) {
        console.error("Error saving data: ", error);
        alert("Terjadi kesalahan saat menyimpan data.");
    }
});

import { getStorage, ref, uploadBytes, getDownloadURL } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-storage.js";

const storage = getStorage();

document.getElementById('profile-pic').addEventListener('change', async (event) => {
    const file = event.target.files[0]; // Ambil file yang dipilih
    if (!file) return;

    const user = auth.currentUser;
    if (!user) {
        alert("Anda harus login untuk mengunggah foto.");
        return;
    }

    const storageRef = ref(storage, `profilePictures/${user.uid}`); // Path penyimpanan di Storage
    try {
        // Upload file ke Firebase Storage
        await uploadBytes(storageRef, file);

        // Ambil URL file yang diunggah
        const photoURL = await getDownloadURL(storageRef);

        // Tampilkan gambar di halaman
        const profileImg = document.getElementById('profile-img');
        profileImg.src = photoURL;
        profileImg.style.display = 'block'; // Tampilkan gambar
        profileImg.style.borderRadius = '50%'; // Buat gambar berbentuk lingkaran

        // Simpan URL foto di Firestore
        const docRef = doc(db, "users", user.uid);
        await setDoc(docRef, { photoURL: photoURL }, { merge: true });

        alert("Foto profil berhasil diunggah dan disimpan!");
    } catch (error) {
        console.error("Error uploading file: ", error);
        alert("Gagal mengunggah foto profil.");
    }
});

onAuthStateChanged(auth, (user) => {
    if (user) {
        const docRef = doc(db, "users", user.uid);

        // Ambil data pengguna dari Firestore
        getDoc(docRef).then((docSnap) => {
            if (docSnap.exists()) {
                const userData = docSnap.data();

                // Isi kolom formulir dengan data Firestore
                document.getElementById('name').innerText = `${userData.firstName || ''} ${userData.lastName || ''}`;
                document.getElementById('phone').value = userData.phone || '';
                document.getElementById('email').value = userData.email || '';
                document.getElementById('address').value = userData.address || '';
                document.getElementById('birth-place').value = userData.birthPlace || '';
                document.getElementById('birth-date').value = userData.birthDate || '';
                document.getElementById('age').value = userData.age || '';

                // Tampilkan foto profil dari URL di Firestore
                if (userData.photoURL) {
                    const profileImg = document.getElementById('profile-img');
                    profileImg.src = userData.photoURL;
                    profileImg.style.display = 'block'; // Tampilkan gambar
                    profileImg.style.borderRadius = '50%'; // Lingkaran
                }
            }
        }).catch((error) => {
            console.error("Error getting document:", error);
        });

        // Ambil URL foto dari Firebase Storage jika belum ada di Firestore
        const storageRef = ref(storage, `profilePictures/${user.uid}`);
        getDownloadURL(storageRef).then((url) => {
            const profileImg = document.getElementById('profile-img');
            profileImg.src = url;
            profileImg.style.display = 'block';
            profileImg.style.borderRadius = '50%';

            // Simpan URL ke Firestore jika belum tersimpan
            const docRef = doc(db, "users", user.uid);
            setDoc(docRef, { photoURL: url }, { merge: true });
        }).catch((error) => {
            console.error("Error getting profile picture URL:", error);
        });
    } else {
        console.log("No user is signed in.");
    }
});
