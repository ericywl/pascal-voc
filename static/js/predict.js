window.onload = function() {
    document.getElementById("files").onchange = function() {
        const reader = new FileReader();
        reader.onload = function(e) {
            // Get loaded data and render thumbnail.
            document.getElementById("image").src = e.target.result;
        };
        // Read the image file as a data URL.
        reader.readAsDataURL(this.files[0]);
        document.getElementById("files-label").innerText = this.files[0].name;
        document.getElementById("predict-btn").disabled = false;
    };

    document.getElementById("predict-btn").onclick = function() {
        file = document.getElementById("files").files[0];
        // Check file type is image
        if (file.type.indexOf("image") < 0) {
            console.log("File must be an image!");
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            const imgBlob = e.target.result;
            const json = { name: file.name, data: imgBlob };
            makePostRequest("/predict", json)
                .then(function(response) {
                    console.log(response);
                })
                .catch(function(error) {
                    console.log(error);
                });
        };
        reader.readAsDataURL(file);
    };
};

const makePostRequest = function(url, json) {
    const xhr = new XMLHttpRequest();
    return new Promise(function(resolve, reject) {
        xhr.onreadystatechange = function() {
            // Request not ready
            if (xhr.readyState !== 4) return;
            // Process response
            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(xhr);
            } else {
                reject({
                    status: xhr.status,
                    statusText: xhr.statusText
                });
            }
        };

        xhr.open("POST", url, true);
        xhr.setRequestHeader("Content-Type", "application/json");
        xhr.send(JSON.stringify(json));
    });
};
