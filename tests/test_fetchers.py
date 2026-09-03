"""Offline tests for museum fetcher parsing (no network)."""

import unittest

from src.dataset.fetchers.npm_tw import parse_detail_metadata, parse_search_ids, iter_canvases, build_image_url, extract_image_codes


SEARCH_HTML = """
<a href="/opendata/Pub/Detail/22?dep=P&amp;mode=full">a</a>
<a href="/opendata/Pub/Detail/10?dep=P&amp;mode=full">b</a>
<a href="/opendata/Pub/Detail/22?dep=P&amp;mode=full">dup</a>
"""

DETAIL_HTML = """
<table>
<tr>
    <td>文物統一編號</td>
    <td>
        故畫000001N000000000
    </td>
</tr>
<tr>
    <td>品名</td>
    <td>
六朝梁張僧繇雪山紅樹圖　軸                                            <br />
Snowy Mountains and Red Trees                                </td>
</tr>
<tr>
    <td>作者</td>
    <td>張僧繇,明人 Anonymous,Ming Dynasty</td>
</tr>
</table>
"""

MANIFEST = {
    "label": "NPM 故宮 - 雪山紅樹圖 軸",
    "sequences": [{
        "canvases": [
            {"label": "K2A000001N000000000PAA", "width": "1100", "height": "2085",
             "images": [{"resource": {"service": {"@id": "https://iiifod.npm.gov.tw/iiif/2/K2A%2FK2A000001N000000000PAA"}}}]},
            {"label": "NOIMG", "images": []},
        ]
    }],
}


class TestNpmTwParsing(unittest.TestCase):
    def test_parse_search_ids_dedup_ordered(self):
        self.assertEqual(parse_search_ids(SEARCH_HTML), ["22", "10"])

    def test_parse_detail_metadata(self):
        meta = parse_detail_metadata(DETAIL_HTML)
        self.assertEqual(meta["文物統一編號"], "故畫000001N000000000")
        self.assertIn("雪山紅樹圖", meta["品名"])
        self.assertIn("Snowy Mountains and Red Trees", meta["品名"])
        self.assertEqual(meta["作者"], "張僧繇,明人 Anonymous,Ming Dynasty")

    def test_iter_canvases_skips_imageless(self):
        canvases = list(iter_canvases(MANIFEST))
        self.assertEqual(len(canvases), 1)
        self.assertEqual(canvases[0]["label"], "K2A000001N000000000PAA")
        self.assertEqual(
            build_image_url(canvases[0]["iiif_service"], ",1600"),
            "https://iiifod.npm.gov.tw/iiif/2/K2A%2FK2A000001N000000000PAA/full/,1600/0/default.jpg",
        )

    def test_extract_image_codes_dedup_ordered(self):
        html = ('<img data-image-name="K2A003652N000000000PAB" />'
                '<img data-image-name="K2A003652N000000000PAC" />'
                '<img data-image-name="K2A003652N000000000PAB" />')
        self.assertEqual(extract_image_codes(html),
                         ["K2A003652N000000000PAB", "K2A003652N000000000PAC"])
        self.assertEqual(extract_image_codes("<html>no images</html>"), [])


if __name__ == "__main__":
    unittest.main()
