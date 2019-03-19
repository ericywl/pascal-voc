window.onload = function() {
    const columns = ["table-id", "table-class", "table-avg-precision"];
    const thead = document.getElementsByTagName("thead")[0];
    thead.onclick = function(e) {
        const index = columns.indexOf(e.target.id);
        if (index < 0 || index > 2) return;
        const sortDir = sortTable(index);
        for (let el of thead.getElementsByTagName("th")) {
            el.getElementsByTagName("span")[0].innerHTML = "";
        }
        e.target.getElementsByTagName("span")[0].innerHTML =
            sortDir === "desc" ? "&#9660;" : "&#9650;";
    };

    const tbody = document.getElementsByTagName("tbody")[0];
    tbody.onclick = function(e) {
        tds = e.target.parentElement.getElementsByTagName("td");
        if (tds.length < 1 || tds.length > 2) return;
        className = tds[0].innerText;
        window.location = "/ranks/" + className;
    };
};

const sortTable = function(n) {
    let table,
        rows,
        switching,
        i,
        x,
        y,
        shouldSwitch,
        dir,
        switchcount = 0;
    table = document.getElementById("ranks-table");
    switching = true;
    // Set the sorting direction to ascending:
    dir = "asc";
    /* Make a loop that will continue until
    no switching has been done: */
    while (switching) {
        // Start by saying: no switching is done:
        switching = false;
        rows = table.rows;
        /* Loop through all table rows (except the
        first, which contains table headers): */
        for (i = 1; i < rows.length - 1; i++) {
            // Start by saying there should be no switching:
            shouldSwitch = false;
            /* Get the two elements you want to compare,
            one from current row and one from the next: */
            if (n === 0) {
                x = rows[i].getElementsByTagName("th")[0];
                y = rows[i + 1].getElementsByTagName("th")[0];
            } else {
                x = rows[i].getElementsByTagName("td")[n - 1];
                y = rows[i + 1].getElementsByTagName("td")[n - 1];
            }
            /* Check if the two rows should switch place,
            based on the direction, asc or desc: */
            if (dir == "asc") {
                if (n === 0) {
                    if (parseInt(x.innerHTML) > parseInt(y.innerHTML)) {
                        shouldSwitch = true;
                        break;
                    }
                } else {
                    if (
                        x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()
                    ) {
                        // If so, mark as a switch and break the loop:
                        shouldSwitch = true;
                        break;
                    }
                }
            } else if (dir == "desc") {
                if (n === 0) {
                    if (parseInt(x.innerHTML) < parseInt(y.innerHTML)) {
                        shouldSwitch = true;
                        break;
                    }
                } else {
                    if (
                        x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()
                    ) {
                        // If so, mark as a switch and break the loop:
                        shouldSwitch = true;
                        break;
                    }
                }
            }
        }
        if (shouldSwitch) {
            /* If a switch has been marked, make the switch
            and mark that a switch has been done: */
            rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
            switching = true;
            // Each time a switch is done, increase this count by 1:
            switchcount++;
        } else {
            /* If no switching has been done AND the direction is "asc",
            set the direction to "desc" and run the while loop again. */
            if (switchcount == 0 && dir == "asc") {
                dir = "desc";
                switching = true;
            }
        }
    }

    return dir;
};
