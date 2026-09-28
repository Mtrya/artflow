"""Offline tests for the reinforcement fetchers (mocked HTTP, no live calls)."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from scripts.data import fetch_commons, fetch_inat, fetch_museum
from scripts.data.fetch_reinforce_common import (
    long_side_bounded,
    parse_queries,
    shape_ok,
    strip_html,
    term_budget,
)

BASE_KEYS = {
    "source", "source_id", "photo_id", "page_url", "original_url", "alt",
    "photographer", "photographer_url", "source_width", "source_height",
    "query", "keep_hint", "license", "title", "artist",
}
SUCCESS_KEYS = BASE_KEYS | {"image_url", "local_path", "width", "height", "download_ok"}
SKIP_KEYS = BASE_KEYS | {"download_ok", "skip_reason"}
FAILED_KEYS = BASE_KEYS | {"image_url", "local_path", "download_ok", "skip_reason"}

INAT_SEARCH = fetch_inat.API_ROOT
MET_SEARCH = fetch_museum.MET_SEARCH
MET_OBJECT_PREFIX = fetch_museum.MET_OBJECT + "/"
AIC_SEARCH = fetch_museum.AIC_SEARCH
COMMONS_API = fetch_commons.API


def jpeg_bytes(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (120, 90, 60)).save(buffer, "JPEG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, payload=None, content=b"", status=200):
        self._payload = payload
        self.content = content
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Routes API json by URL prefix and images by URL substring, recording calls."""

    def __init__(self, json_routes=None, images=None):
        self.json_routes = dict(json_routes or {})
        self.images = dict(images or {})
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        for prefix, route in self.json_routes.items():
            if url.startswith(prefix):
                payload = route(url, dict(params or {})) if callable(route) else route
                return payload if isinstance(payload, FakeResponse) else FakeResponse(payload)
        for substring, value in self.images.items():
            if substring in url:
                return value if isinstance(value, FakeResponse) else FakeResponse(content=jpeg_bytes(*value))
        raise AssertionError(f"unexpected request: {url} {params}")

    def image_calls(self, substring):
        return [call for call in self.calls if substring in call[0]]

    def json_calls(self, prefix):
        return [call for call in self.calls if call[0].startswith(prefix)]


class ReinforceTest(unittest.TestCase):
    """Base whose temporary directory lives until the test method ends."""

    def tmpdir(self):
        return Path(self.enterContext(tempfile.TemporaryDirectory()))


def make_queries(out, text):
    path = Path(out) / "queries.txt"
    path.write_text(text, encoding="utf-8")
    return path


def options(module, out, queries, **overrides):
    flags = {"target": 100, "per_hour": 1000000, "workers": 2}
    flags.update(overrides)
    argv = ["--out", str(out), "--queries", str(queries)]
    for key, value in flags.items():
        argv += ["--" + key.replace("_", "-"), str(value)]
    return module.build_parser().parse_args(argv)


def run_harvest(module, args, session):
    with mock.patch.object(module, "thread_session", return_value=session):
        module.harvest(args, session)


def read_metadata(out):
    path = Path(out) / "metadata.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_state(out):
    return json.loads((Path(out) / "state.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- iNaturalist

def inat_photo(photo_id, license_code="cc-by", width=2048, height=1536):
    return {
        "id": photo_id,
        "license_code": license_code,
        "original_dimensions": {"width": width, "height": height},
        "url": f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/square.jpg",
        "attribution": f"(c) observer{photo_id}, some rights reserved (CC BY-NC)",
    }


def inat_observation(observation_id, name, common, photos):
    return {
        "id": observation_id,
        "uri": f"https://www.inaturalist.org/observations/{observation_id}",
        "taxon": {"name": name, "preferred_common_name": common} if name else None,
        "photos": photos,
    }


def inat_payload(page, per_page, total, results):
    return {"total_results": total, "page": page, "per_page": per_page, "results": results}


def inat_routes(pages, total=400):
    return {INAT_SEARCH: lambda url, params: inat_payload(
        params["page"], params["per_page"], total, pages.get(params["page"], []))}


class TestSharedHelpers(unittest.TestCase):

    def test_parse_queries_caps_and_comments(self):
        text = "# a comment\n\nhanfu dress\nkimono | 25\nsari |\n"
        self.assertEqual(parse_queries(text),
                         [("hanfu dress", None), ("kimono", 25), ("sari", None)])

    def test_term_budget(self):
        self.assertIsNone(term_budget(None, 12))
        self.assertEqual(term_budget(10, 4), 6)
        self.assertEqual(term_budget(10, 12), 0)

    def test_shape_filter_boundaries(self):
        self.assertTrue(shape_ok(4000, 3000)[0])
        self.assertEqual(long_side_bounded(4000, 3000), (1792, 1344))
        # exactly 2:1 and exactly 896 px short side pass
        self.assertTrue(shape_ok(2000, 1000)[0])
        self.assertFalse(shape_ok(2000, 950)[0])
        self.assertFalse(shape_ok(3000, 1000)[0])
        self.assertFalse(shape_ok(800, 600)[0])
        self.assertFalse(shape_ok(0, 0)[0])

    def test_strip_html(self):
        self.assertEqual(strip_html("<p>A stone <b>bridge</b></p> &amp; river"),
                         "A stone bridge & river")


class TestInat(ReinforceTest):

    def test_record_and_download(self):
        observation = inat_observation(101, "Macaca mulatta", "Rhesus Macaque",
                                       [inat_photo(7)])
        session = FakeSession(json_routes=inat_routes({1: [observation]}),
                              images={"photos/7": (2048, 1536)})
        out = self.tmpdir()
        args = options(fetch_inat, out, make_queries(out, "monkey | 5"), per_page=200)
        run_harvest(fetch_inat, args, session)
        rows = read_metadata(out)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["source"], "reinforce_inat")
        self.assertEqual(row["source_id"], "inat-7")
        self.assertEqual(row["photo_id"], 7)
        self.assertEqual(row["image_url"],
                         "https://inaturalist-open-data.s3.amazonaws.com/photos/7/original.jpg")
        self.assertEqual(row["original_url"], row["image_url"])
        self.assertEqual(row["alt"], "Macaca mulatta (Rhesus Macaque)")
        self.assertEqual(row["license"], "cc-by")
        self.assertIn("observer7", row["photographer"])
        self.assertEqual(row["page_url"], "https://www.inaturalist.org/observations/101")
        self.assertEqual((row["source_width"], row["source_height"]), (2048, 1536))
        # saved image is bounded to the 1792 px box, like the other sources
        self.assertEqual((row["width"], row["height"]), (1792, 1344))
        self.assertTrue(row["download_ok"])
        self.assertTrue(row["keep_hint"])
        self.assertEqual(row["title"], "")
        self.assertTrue(Path(row["local_path"]).exists())
        params = session.calls[0][1]
        self.assertEqual(params["taxon_name"], "monkey")
        self.assertEqual(params["photo_license"], "cc0,cc-by,cc-by-sa,cc-by-nc")
        self.assertEqual(params["quality_grade"], "research")
        self.assertEqual(params["photos"], "true")
        self.assertEqual(params["page"], 1)

    def test_license_and_null_taxon_filters(self):
        observation = inat_observation(
            102, "Vulpes vulpes", "Red Fox",
            [inat_photo(8, license_code="cc-by-nc-nd"), inat_photo(9, license_code="cc0")])
        no_taxon = inat_observation(103, None, None, [inat_photo(10)])
        session = FakeSession(json_routes=inat_routes({1: [observation, no_taxon]}),
                              images={"photos/9": (1200, 900)})
        out = self.tmpdir()
        args = options(fetch_inat, out, make_queries(out, "fox | 5"))
        run_harvest(fetch_inat, args, session)
        rows = read_metadata(out)
        self.assertEqual(len(rows), 2)
        skipped, kept = rows
        self.assertFalse(skipped["download_ok"])
        self.assertEqual(skipped["skip_reason"], "license cc-by-nc-nd")
        self.assertEqual(skipped["source_id"], "inat-8")
        self.assertTrue(kept["download_ok"])
        self.assertEqual(kept["source_id"], "inat-9")
        self.assertEqual(session.image_calls("photos/8"), [])
        self.assertEqual(session.image_calls("photos/10"), [])
        self.assertEqual(len(session.image_calls("photos/9")), 1)

    def test_shape_filter_pre_download(self):
        observation = inat_observation(104, "Pica pica", "Eurasian Magpie",
                                       [inat_photo(11, width=3000, height=1000)])
        session = FakeSession(json_routes=inat_routes({1: [observation]}))
        out = self.tmpdir()
        args = options(fetch_inat, out, make_queries(out, "magpie | 5"))
        run_harvest(fetch_inat, args, session)
        rows = read_metadata(out)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["download_ok"])
        self.assertEqual(rows[0]["skip_reason"], "shape 3000x1000")
        self.assertEqual(session.image_calls("photos/11"), [])

    def test_cap_exhausts_term_without_walking_more_pages(self):
        photos = [inat_photo(photo_id) for photo_id in (12, 13, 14)]
        observation = inat_observation(105, "Sciurus vulgaris", "Red Squirrel", photos)
        session = FakeSession(json_routes=inat_routes({1: [observation]}, total=400),
                              images={"photos/12": (1600, 1200), "photos/13": (1600, 1200)})
        out = self.tmpdir()
        args = options(fetch_inat, out, make_queries(out, "squirrel | 2"))
        run_harvest(fetch_inat, args, session)
        rows = read_metadata(out)
        state = read_state(out)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["download_ok"] for row in rows))
        self.assertTrue(state["squirrel"]["exhausted"])
        self.assertEqual(len(session.json_calls(INAT_SEARCH)), 1)
        self.assertEqual(session.json_calls(INAT_SEARCH)[0][1]["page"], 1)

    def test_resume_skips_fulfilled_terms_and_recorded_ids(self):
        monkey = inat_observation(106, "Macaca mulatta", "Rhesus Macaque", [inat_photo(20)])
        fox = inat_observation(107, "Vulpes vulpes", "Red Fox", [inat_photo(30)])

        def route(url, params):
            found = monkey if params["taxon_name"] == "monkey" else fox
            return inat_payload(params["page"], params["per_page"], 1, [found])

        out = self.tmpdir()
        queries = make_queries(out, "monkey | 1\nfox | 1")
        session_one = FakeSession(json_routes={INAT_SEARCH: route},
                                  images={"photos/20": (1200, 900),
                                          "photos/30": (1200, 900)})
        run_harvest(fetch_inat, options(fetch_inat, out, queries), session_one)
        self.assertEqual(len(read_metadata(out)), 2)
        second = FakeSession()
        run_harvest(fetch_inat, options(fetch_inat, out, queries), second)
        self.assertEqual(second.calls, [])
        self.assertEqual(len(read_metadata(out)), 2)

    def test_partial_term_resumes_from_state_page(self):
        first = inat_observation(108, "Lynx lynx", "Eurasian Lynx", [inat_photo(40)])
        second = inat_observation(109, "Lynx lynx", "Eurasian Lynx", [inat_photo(41)])
        out = self.tmpdir()
        queries = make_queries(out, "lynx | 5")
        session_one = FakeSession(json_routes=inat_routes({1: [first], 2: [second]}),
                                  images={"photos/40": (1200, 900)})
        run_harvest(fetch_inat, options(fetch_inat, out, queries, target=1), session_one)
        state = read_state(out)
        self.assertEqual(state["lynx"]["page"], 2)
        self.assertFalse(state["lynx"]["exhausted"])
        session_two = FakeSession(json_routes=inat_routes({2: [second]}),
                                  images={"photos/41": (1200, 900)})
        run_harvest(fetch_inat, options(fetch_inat, out, queries, target=2), session_two)
        rows = read_metadata(out)
        state = read_state(out)
        self.assertEqual([params["page"] for _, params in session_two.json_calls(INAT_SEARCH)], [2])
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["source_id"] for row in rows}, {"inat-40", "inat-41"})
        self.assertTrue(state["lynx"]["exhausted"])


# ---------------------------------------------------------------- Met and AIC

def met_object(object_id, primary=True, small=False, title="Chair", artist="Anon"):
    return {
        "objectID": object_id,
        "title": title,
        "artistDisplayName": artist,
        "objectURL": f"https://www.metmuseum.org/art/collection/search/{object_id}",
        "primaryImage": f"https://images.metmuseum.org/{object_id}-full.jpg" if primary else "",
        "primaryImageSmall": (f"https://images.metmuseum.org/{object_id}-small.jpg"
                              if primary or small else ""),
    }


def met_routes(object_ids, objects):
    return {
        MET_SEARCH: lambda url, params: {"total": len(object_ids), "objectIDs": object_ids},
        MET_OBJECT_PREFIX: lambda url, params: objects[int(url.rsplit("/", 1)[1])],
    }


def aic_entry(artwork_id, title="Samovar", image_id="uuid", public=True, artist="Anon"):
    return {"id": artwork_id, "title": title, "image_id": image_id,
            "artist_title": artist, "is_public_domain": public}


def aic_payload(data, total_pages=1, iiif="https://iiif.example.org/2"):
    payload = {"data": data, "pagination": {"total_pages": total_pages, "current_page": 1}}
    if iiif:
        payload["config"] = {"iiif_url": iiif}
    return payload


class TestMuseum(ReinforceTest):

    def test_parse_apis_order_and_validation(self):
        self.assertEqual(fetch_museum.parse_apis("met,aic"), ["met", "aic"])
        self.assertEqual(fetch_museum.parse_apis("aic,met"), ["met", "aic"])
        self.assertEqual(fetch_museum.parse_apis("aic"), ["aic"])
        with self.assertRaises(SystemExit):
            fetch_museum.parse_apis("rijksmuseum")

    def test_met_filters_and_local_bounding(self):
        objects = {1: met_object(1, primary=False, small=True),
                   2: met_object(2, title="Samovar", artist="Henry van de Velde"),
                   3: met_object(3, primary=False, small=False)}
        session = FakeSession(json_routes=met_routes([1, 2, 3], objects),
                              images={"2-full.jpg": (2500, 1500)})
        out = self.tmpdir()
        args = options(fetch_museum, out, make_queries(out, "samovar | 5"), apis="met")
        run_harvest(fetch_museum, args, session)
        rows = read_metadata(out)
        state = read_state(out)
        self.assertEqual(len(rows), 3)
        small_only, kept, imageless = rows
        self.assertEqual(small_only["skip_reason"], "primaryImageSmall only")
        self.assertEqual(imageless["skip_reason"], "no primaryImage")
        self.assertTrue(kept["download_ok"])
        self.assertEqual(kept["source"], "reinforce_museum")
        self.assertEqual(kept["source_id"], "met-2")
        self.assertEqual(kept["photo_id"], 2)
        self.assertEqual(kept["title"], "Samovar")
        self.assertEqual(kept["artist"], "Henry van de Velde")
        self.assertEqual(kept["photographer"], "Henry van de Velde")
        self.assertEqual(kept["page_url"], "https://www.metmuseum.org/art/collection/search/2")
        self.assertEqual(kept["license"], "CC0 (Met Open Access)")
        self.assertEqual(kept["alt"], "Samovar")
        self.assertEqual((kept["width"], kept["height"]), (1792, 1075))
        self.assertEqual((kept["source_width"], kept["source_height"]), (2500, 1500))
        self.assertTrue(Path(kept["local_path"]).exists())
        image_calls = session.image_calls("-full.jpg")
        self.assertEqual(len(image_calls), 1)
        self.assertIn("2-full.jpg", image_calls[0][0])
        self.assertEqual(session.json_calls(MET_SEARCH)[0][1],
                         {"hasImages": "true", "q": "samovar"})
        self.assertTrue(state["samovar"]["met_done"])
        self.assertTrue(state["samovar"]["exhausted"])

    def test_met_download_failure_row(self):
        session = FakeSession(json_routes=met_routes([5], {5: met_object(5)}),
                              images={"5-full.jpg": FakeResponse(status=500)})
        out = self.tmpdir()
        args = options(fetch_museum, out, make_queries(out, "chair | 5"), apis="met")
        run_harvest(fetch_museum, args, session)
        rows = read_metadata(out)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["download_ok"])
        self.assertEqual(rows[0]["skip_reason"], "download failed")
        self.assertEqual(set(rows[0]), FAILED_KEYS)

    def test_aic_filters_and_iiif_url(self):
        entries = [aic_entry(10, image_id=None),
                   aic_entry(11, public=False),
                   aic_entry(12, image_id="good-uuid", title="Chair")]
        session = FakeSession(json_routes={AIC_SEARCH: aic_payload(entries)},
                              images={"iiif.example.org/2/good-uuid": (1600, 1200)})
        out = self.tmpdir()
        args = options(fetch_museum, out, make_queries(out, "chair | 5"), apis="aic")
        run_harvest(fetch_museum, args, session)
        rows = read_metadata(out)
        state = read_state(out)
        self.assertEqual(len(rows), 3)
        no_image, not_public, kept = rows
        self.assertEqual(no_image["skip_reason"], "no image_id")
        self.assertEqual(not_public["skip_reason"], "not public domain")
        self.assertTrue(kept["download_ok"])
        self.assertEqual(kept["source_id"], "aic-12")
        self.assertEqual(kept["image_url"],
                         "https://iiif.example.org/2/good-uuid/full/!1792,1792/0/default.jpg")
        self.assertEqual(kept["page_url"], "https://www.artic.edu/artworks/12")
        self.assertEqual(kept["license"], "CC0 (Art Institute of Chicago)")
        self.assertEqual((kept["width"], kept["height"]), (1600, 1200))
        self.assertEqual(len(session.image_calls("good-uuid")), 1)
        params = session.json_calls(AIC_SEARCH)[0][1]
        self.assertEqual(params["q"], "chair")
        self.assertEqual(params["limit"], 100)
        self.assertEqual(params["fields"], "id,title,image_id,artist_title,is_public_domain")
        self.assertTrue(state["chair"]["aic_done"])

    def test_aic_falls_back_to_default_iiif_prefix(self):
        # no config in the response: the script's own prefix is used
        session = FakeSession(json_routes={AIC_SEARCH: aic_payload(
            [aic_entry(13, image_id="plain-uuid")], iiif=None)},
            images={"artic.edu/iiif/2/plain-uuid": (1500, 1500)})
        out = self.tmpdir()
        args = options(fetch_museum, out, make_queries(out, "chair | 5"), apis="aic")
        run_harvest(fetch_museum, args, session)
        rows = read_metadata(out)
        self.assertEqual(rows[0]["image_url"],
                         "https://www.artic.edu/iiif/2/plain-uuid/full/!1792,1792/0/default.jpg")
        self.assertEqual(fetch_museum.aic_image_url("https://www.artic.edu/iiif/2", "x"),
                         "https://www.artic.edu/iiif/2/x/full/!1792,1792/0/default.jpg")

    def test_cap_shared_across_met_and_aic(self):
        session = FakeSession(
            json_routes={**met_routes([7], {7: met_object(7)}),
                         AIC_SEARCH: aic_payload(
                             [aic_entry(20, image_id="first"),
                              aic_entry(21, image_id="second")])},
            images={"7-full.jpg": (2000, 2000), "iiif.example.org/2/first": (1500, 1500)})
        out = self.tmpdir()
        args = options(fetch_museum, out, make_queries(out, "chair | 2"))
        run_harvest(fetch_museum, args, session)
        rows = read_metadata(out)
        state = read_state(out)
        kept = [row for row in rows if row["download_ok"]]
        self.assertEqual(len(kept), 2)
        self.assertEqual({row["source_id"] for row in kept}, {"met-7", "aic-20"})
        self.assertEqual(len(session.json_calls(MET_SEARCH)), 1)
        self.assertEqual(len(session.json_calls(AIC_SEARCH)), 1)
        self.assertTrue(state["chair"]["exhausted"])

    def test_resume_skips_fulfilled_term(self):
        out = self.tmpdir()
        queries = make_queries(out, "chair | 1")
        session_one = FakeSession(json_routes=met_routes([9], {9: met_object(9)}),
                                  images={"9-full.jpg": (1200, 900)})
        run_harvest(fetch_museum, options(fetch_museum, out, queries, apis="met"),
                    session_one)
        self.assertEqual(len(read_metadata(out)), 1)
        second = FakeSession()
        run_harvest(fetch_museum, options(fetch_museum, out, queries, apis="met"), second)
        self.assertEqual(second.calls, [])
        self.assertEqual(len(read_metadata(out)), 1)


# ---------------------------------------------------------------- Commons


def commons_entry(page_id, title="Stone bridge.jpg", mime="image/jpeg",
                  width=3000, height=2000,
                  description="<p>A stone <b>bridge</b> over a river</p>",
                  artist="<a href='#'>Jane Doe</a>", license_name="CC BY-SA 4.0",
                  index=1):
    extmetadata = {}
    if description is not None:
        extmetadata["ImageDescription"] = {"value": description}
    if artist is not None:
        extmetadata["Artist"] = {"value": artist}
    if license_name is not None:
        extmetadata["LicenseShortName"] = {"value": license_name}
    return {
        "pageid": page_id, "ns": 6, "title": f"File:{title}", "index": index,
        "imageinfo": [{
            "url": f"https://upload.wikimedia.org/original/{page_id}.jpg",
            "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{title.replace(' ', '_')}",
            "thumburl": f"https://upload.wikimedia.org/thumb/{page_id}/1792px.jpg",
            "width": width, "height": height, "mime": mime,
            "extmetadata": extmetadata,
        }],
    }


def commons_payload(entries, cont=None):
    payload = {"query": {"pages": {str(entry["pageid"]): entry for entry in entries}}}
    if cont:
        payload["continue"] = cont
    return payload


def commons_routes(pages):
    """``pages`` maps a gsrcontinue token (None for the first request) to entries."""
    def route(url, params):
        value = pages[params.get("gsrcontinue")]
        return commons_payload(value) if isinstance(value, list) else value
    return {COMMONS_API: route}


class TestCommons(ReinforceTest):

    def test_mime_filter_blocks_non_jpeg(self):
        entry = commons_entry(1, title="Bridge diagram.png", mime="image/png")
        session = FakeSession(json_routes=commons_routes({None: [entry]}))
        out = self.tmpdir()
        args = options(fetch_commons, out, make_queries(out, "bridge | 5"))
        run_harvest(fetch_commons, args, session)
        rows = read_metadata(out)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["download_ok"])
        self.assertEqual(rows[0]["skip_reason"], "mime image/png")
        self.assertEqual(rows[0]["source_id"], "com-1")
        self.assertEqual(session.image_calls("1792px"), [])

    def test_shape_filter_pre_download(self):
        entry = commons_entry(2, width=2500, height=800)
        session = FakeSession(json_routes=commons_routes({None: [entry]}))
        out = self.tmpdir()
        args = options(fetch_commons, out, make_queries(out, "bridge | 5"))
        run_harvest(fetch_commons, args, session)
        rows = read_metadata(out)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["skip_reason"], "shape 2500x800")
        self.assertEqual(session.image_calls("1792px"), [])

    def test_record_fields_and_description_fallback(self):
        entries = [commons_entry(3), commons_entry(4, title="Old mill.jpg", description=None)]
        session = FakeSession(json_routes=commons_routes({None: entries}),
                              images={"thumb/3": (1792, 1195), "thumb/4": (1792, 1195)})
        out = self.tmpdir()
        args = options(fetch_commons, out, make_queries(out, "bridge | 5"))
        run_harvest(fetch_commons, args, session)
        rows = read_metadata(out)
        self.assertEqual(len(rows), 2)
        by_id = {row["source_id"]: row for row in rows}
        described, fallback = by_id["com-3"], by_id["com-4"]
        self.assertEqual(described["source"], "reinforce_commons")
        self.assertEqual(described["source_id"], "com-3")
        self.assertEqual(described["photo_id"], 3)
        self.assertEqual(described["alt"], "A stone bridge over a river")
        self.assertEqual(described["title"], "Stone bridge")
        self.assertEqual(described["artist"], "Jane Doe")
        self.assertEqual(described["photographer"], "Jane Doe")
        self.assertEqual(described["license"], "CC BY-SA 4.0")
        self.assertEqual(described["page_url"],
                         "https://commons.wikimedia.org/wiki/File:Stone_bridge.jpg")
        self.assertEqual(described["original_url"],
                         "https://upload.wikimedia.org/original/3.jpg")
        self.assertEqual(described["image_url"],
                         "https://upload.wikimedia.org/thumb/3/1792px.jpg")
        self.assertEqual((described["source_width"], described["source_height"]),
                         (3000, 2000))
        self.assertEqual((described["width"], described["height"]), (1792, 1195))
        self.assertTrue(Path(described["local_path"]).exists())
        self.assertEqual(fallback["alt"], "Old mill")
        self.assertEqual(fallback["title"], "Old mill")

    def test_title_cleaning(self):
        self.assertEqual(fetch_commons.clean_title("File:Stone_bridge.jpg"), "Stone bridge")
        self.assertEqual(fetch_commons.clean_title("File:Map.svg"), "Map")
        self.assertEqual(fetch_commons.clean_title("File:Weird.picture.jpg"), "Weird.picture")

    def test_gsrcontinue_pagination_then_resume(self):
        entries = [commons_entry(5), commons_entry(6, title="Old mill.jpg")]
        continuation = {"gsrcontinue": "tok", "continue": "-||"}
        session = FakeSession(
            json_routes=commons_routes({None: commons_payload([entries[0]], continuation),
                                        "tok": commons_payload([entries[1]])}),
            images={"thumb/5": (1792, 1195), "thumb/6": (1792, 1195)})
        out = self.tmpdir()
        queries = make_queries(out, "bridge")
        run_harvest(fetch_commons, options(fetch_commons, out, queries), session)
        rows = read_metadata(out)
        state = read_state(out)
        self.assertEqual(len(rows), 2)
        self.assertTrue(state["bridge"]["exhausted"])
        first, second_page = session.json_calls(COMMONS_API)
        self.assertEqual(first[1]["gsrsearch"], "filetype:bitmap bridge")
        self.assertEqual(first[1]["gsrnamespace"], 6)
        self.assertEqual(second_page[1]["gsrcontinue"], "tok")
        self.assertEqual(second_page[1]["continue"], "-||")
        fresh = FakeSession()
        run_harvest(fetch_commons, options(fetch_commons, out, queries), fresh)
        self.assertEqual(fresh.calls, [])
        self.assertEqual(len(read_metadata(out)), 2)

    def test_cap_exhausts_term(self):
        entries = [commons_entry(page_id) for page_id in (7, 8, 9)]
        session = FakeSession(
            json_routes=commons_routes({None: commons_payload(
                entries, {"gsrcontinue": "tok"})}),
            images={"thumb/7": (1792, 1195)})
        out = self.tmpdir()
        args = options(fetch_commons, out, make_queries(out, "bridge | 1"))
        run_harvest(fetch_commons, args, session)
        rows = read_metadata(out)
        state = read_state(out)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["download_ok"])
        self.assertTrue(state["bridge"]["exhausted"])
        self.assertEqual(len(session.json_calls(COMMONS_API)), 1)


class TestRecordSchema(ReinforceTest):

    def test_inat_row_keys(self):
        observation = inat_observation(
            200, "Pica pica", "Eurasian Magpie",
            [inat_photo(50, license_code="cc-by-nc-nd"), inat_photo(51, license_code="cc0")])
        session = FakeSession(json_routes=inat_routes({1: [observation]}),
                              images={"photos/51": (1600, 1200)})
        out = self.tmpdir()
        args = options(fetch_inat, out, make_queries(out, "magpie | 5"))
        run_harvest(fetch_inat, args, session)
        rows = read_metadata(out)
        by_id = {row["source_id"]: row for row in rows}
        self.assertEqual(set(by_id["inat-50"]), SKIP_KEYS)
        self.assertEqual(set(by_id["inat-51"]), SUCCESS_KEYS)

    def test_museum_row_keys(self):
        session = FakeSession(json_routes=met_routes([1], {1: met_object(1)}),
                              images={"1-full.jpg": (1600, 1200)})
        out = self.tmpdir()
        args = options(fetch_museum, out, make_queries(out, "chair | 5"), apis="met")
        run_harvest(fetch_museum, args, session)
        rows = read_metadata(out)
        self.assertEqual(set(rows[0]), SUCCESS_KEYS)

    def test_commons_row_keys(self):
        entries = [commons_entry(30), commons_entry(31, mime="image/png")]
        session = FakeSession(json_routes=commons_routes({None: entries}),
                              images={"thumb/30": (1792, 1195)})
        out = self.tmpdir()
        args = options(fetch_commons, out, make_queries(out, "bridge | 5"))
        run_harvest(fetch_commons, args, session)
        rows = read_metadata(out)
        by_id = {row["source_id"]: row for row in rows}
        self.assertEqual(set(by_id["com-30"]), SUCCESS_KEYS)
        self.assertEqual(set(by_id["com-31"]), SKIP_KEYS)


if __name__ == "__main__":
    unittest.main()
