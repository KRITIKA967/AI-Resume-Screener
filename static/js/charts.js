// Fetch chart data from Flask route /chart_data
let ctx = document.getElementById("pieChart");

fetch("/chart_data")
    .then(response => response.json())
    .then(data => {
        new Chart(ctx, {
            type: "pie",
            data: {
                labels: ["Shortlisted", "Not Shortlisted"],
                datasets: [{
                    data: [data.shortlisted, data.not_shortlisted],
                    backgroundColor: ["#28a745", "#dc3545"],
                    borderColor: "#fff",
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: { font: { size: 14 } }
                    }
                }
            }
        });
    })
    .catch(err => console.error("Chart load error:", err));
