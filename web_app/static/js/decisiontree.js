var myModal = new bootstrap.Modal(document.getElementById("exampleModal"));
        myModal.show();

        anychart.onDocumentReady(function () {

            // add data
            var data = anychart.data.set([
                ['NonFraud', 80],
                ['Fraud', 20]
            ]);

            // create a pie chart with the data
            var chart = anychart.pie(data);

            // set the chart radius making a donut chart
            chart.innerRadius('55%')

            // create a color palette
            var palette = anychart.palettes.distinctColors();

            // set the colors according to the brands
            palette.items([
                { color: '#25476A' },
                { color: '#FA9F1B' }
            ]);

            // apply the donut chart color palette
            chart.palette(palette);

            // set the position of labels
            //   chart.labels().format('{%x} — {%y}%').fontSize(12);

            // disable the legend
            //   chart.legend(false);

            // format the donut chart tooltip
            chart.tooltip().format('{%PercentValue}%');

            // create a standalone label
            var label = anychart.standalones.label();

            // configure the label settings
            label
                .useHtml(true)
                .text(
                    '<span style = "color: #313136; font-size:20px; font-weight: bold;">21.480</span>' +
                    '<br/><br/></br><span style="color:#444857; font-size: 14px;"><i>+15</i></span>'
                )
                .position('center')
                .anchor('center')
                .hAlign('center')
                .vAlign('middle');

            // set the label as the center content
            chart.center().content(label);

            // set container id for the chart
            chart.container('container');

            // initiate chart drawing
            chart.draw();

        });