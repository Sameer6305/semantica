document.addEventListener("DOMContentLoaded", function () {
    // Target the header title
    var headerTitle = document.querySelector(".md-header__title");

    if (headerTitle) {
        // Create the container for the version selector
        var versionContainer = document.createElement("div");
        versionContainer.className = "version-scroll-container";

        // Define versions
        var versions = [
            { name: "0.1.1", url: "#", current: true },
            { name: "0.1.0", url: "#", current: false }
        ];

        // Create the scrollable list
        var versionList = document.createElement("div");
        versionList.className = "version-list";

        versions.forEach(function (version) {
            var versionLink = document.createElement("a");
            versionLink.className = "version-tag" + (version.current ? " active" : "");
            versionLink.href = version.url;
            versionLink.textContent = version.name;
            versionList.appendChild(versionLink);
        });

        versionContainer.appendChild(versionList);

        // Insert after the header title
        headerTitle.parentNode.insertBefore(versionContainer, headerTitle.nextSibling);
    }
});
