window.onload = function() {
    const columns = ["table-id", "table-class", "table-avg-precision"];
    const thead = document.getElementsByTagName("thead")[0];
    thead.onclick = function(e) {
        targ = e.target
        if (e.target.tagName.toLowerCase() === "span") {
            targ = e.target.parentElement;
        }
        const index = columns.indexOf(targ.id);
        if (index < 0 || index > 2) return;
        const asc = sortTable(targ);
        for (let el of thead.getElementsByTagName("th")) {
            el.getElementsByTagName("span")[0].innerHTML = "";
        }
        targ.getElementsByTagName("span")[0].innerHTML =
            asc ? "&#9650;" : "&#9660;";
    };

    const tbody = document.getElementsByTagName("tbody")[0];
    tbody.onclick = function(e) {
        tds = e.target.parentElement.getElementsByTagName("td");
        if (tds.length < 1 || tds.length > 2) return;
        className = tds[0].innerText;
        window.location = "/ranks/" + className;
    };
};

/* Table sorting */
const getCellValue = (tr, idx) =>
    tr.children[idx].innerText || tr.children[idx].textContent;

const comparer = (idx, asc) => (a, b) =>
    ((v1, v2) =>
        v1 !== "" && v2 !== "" && !isNaN(v1) && !isNaN(v2)
            ? v1 - v2
            : v1.toString().localeCompare(v2))(
        getCellValue(asc ? a : b, idx),
        getCellValue(asc ? b : a, idx)
    );

const sortTable = th => {
    const table = th.closest("table");
    const tbody = table.querySelector("tbody");
    Array.from(tbody.querySelectorAll("tr"))
        .sort(
            comparer(
                Array.from(th.parentNode.children).indexOf(th),
                (this.asc = !this.asc)
            )
        )
        .forEach(tr => tbody.appendChild(tr));

    return this.asc;
};
