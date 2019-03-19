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
        types = ["image/jpg", "image/jpeg", "image/png"];
        // Check file type is image
        if (types.indexOf(file.type) < 0) {
            console.log("File must be JPEG or PNG!");
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            const imgBlob = e.target.result;
            const json = { name: file.name, data: imgBlob };
            makePostRequest("/predict", json)
                .then(function(resp) {
                    console.log(resp);
                    let preds = document.getElementById("preds")
                    if (preds.classList.contains("d-flex")) {
                        preds.classList.remove("d-flex")
                    }
                    preds.innerHTML = json2Table(
                        JSON.parse(resp.response)
                    );
                    document.getElementById("files").disabled = false;
                })
                .catch(function(error) {
                    console.log(error);
                });
        };
        reader.readAsDataURL(file);
        document.getElementById("files").disabled = true;
    };
};

const makePostRequest = function(url, json) {
    const xhr = new XMLHttpRequest();
    return new Promise(function(resolve, reject) {
        xhr.onreadystatechange = function() {
            // Request not ready
            if (xhr.readyState !== 4) {
                document.getElementById("predict-btn").disabled = true;
                let preds = document.getElementById("preds")
                if (!preds.classList.contains("d-flex")) {
                    preds.classList.add("d-flex")
                }
                preds.innerHTML =
                    "<div id='loading' class='spinner-border text-primary' " +
                    "role='status'><span class='sr-only'>Loading...</span>" +
                    "</div>";
                return;
            }
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

const json2Table = function(json) {
    const orderedJsonArray = Object.keys(json)
        .map(function(key) {
            return [key, json[key]];
        })
        .sort(function(a, b) {
            return b[1] - a[1];
        });

    tableStr =
        "<table class='table' style='margin-top: 10px'>" +
        "<thead><tr>" +
        "<th scope='col'>#</th>" +
        "<th scope='col'>Label</th>" +
        "<th scope='col'>Confidence</th>" +
        "</tr></thead>" +
        "<tbody>";

    for (let i = 0, len = orderedJsonArray.length; i < len; i++) {
        tmp = "<tr><th scope='row'>";
        tmp += (i + 1);
        tmp += "</th>";
        tmp += "<td>";
        tmp += orderedJsonArray[i][0];
        tmp += "</td>";
        tmp += "<td>";
        tmp += (orderedJsonArray[i][1] * 100).toFixed(2) + '%';
        tmp += "</td>";
        tmp += "</tr>";
        tableStr += tmp;
    }
    tableStr += "</tbody></table>";
    return tableStr
};
