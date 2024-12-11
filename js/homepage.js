import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getAuth, onAuthStateChanged, signOut } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-auth.js";
import { getFirestore, getDoc, doc } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-firestore.js";

// For Firebase JS SDK v7.20.0 and later, measurementId is optional
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
        if (!user.emailVerified && !user.providerData.some(provider => provider.providerId === 'google.com')) {
            console.log("Email not verified");
            signOut(auth).then(() => {
                window.location.href = 'index.html';
            }).catch((error) => {
                console.error('Error Signing out:', error);
            });
            return;
        }
        const loggedInUserId = localStorage.getItem('loggedInUserId');
        if (loggedInUserId) {
            console.log(user);
            const docRef = doc(db, "users", loggedInUserId);
            getDoc(docRef)
                .then((docSnap) => {
                    if (docSnap.exists()) {
                        const userData = docSnap.data();
                        document.getElementById('user-name').innerText = `${userData.firstName} ${userData.lastName}`;
                        document.getElementById('user-email').innerText = userData.email;

                        // Display notification
                        const notification = document.getElementById('notification');
                        notification.style.display = 'block';

                        // Hide notification after 5 seconds
                        setTimeout(() => {
                            notification.style.display = 'none';
                        }, 5000);
                    } else {
                        console.log("no document found matching id");
                    }
                })
                .catch((error) => {
                    console.log("Error getting document");
                });
        } else {
            console.log("User Id not Found in Local storage");
        }
    } else {
        alert("You must log in/register first.");
        window.location.href = 'index.html';
    }
});

const logoutButton = document.getElementById('logout');

logoutButton.addEventListener('click', () => {
    localStorage.removeItem('loggedInUserId');
    signOut(auth)
        .then(() => {
            window.location.href = 'dashboard.html';
        })
        .catch((error) => {
            console.error('Error Signing out:', error);
        });
});