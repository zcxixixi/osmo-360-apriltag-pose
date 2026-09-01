from pathlib import Path

import pytest
from defusedxml.common import EntitiesForbidden

from osmo360.visualization.render_dual_camera_alignment_demo import (
    load_urdf_wireframe,
)


def test_urdf_loader_rejects_xml_entity_definitions(tmp_path: Path) -> None:
    urdf = tmp_path / "entity-expansion.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<!DOCTYPE robot [<!ENTITY payload "expanded">]>
<robot name="unsafe"><link name="&payload;" /></robot>
""",
        encoding="utf-8",
    )

    with pytest.raises(EntitiesForbidden):
        load_urdf_wireframe(urdf)
