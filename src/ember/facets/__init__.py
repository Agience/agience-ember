"""The HTML surfaces ember renders: the ontology browser, the mesh dashboard and the status page.

Output is aria's charter, so this package is grouped for extraction to `chorus/aria` as a single
move rather than as a permanent home.

`browse` holds both ontology views and node-status views, and reaches into ember internals for the
latter — access, keyed, content, stats, improve, pool, genesis, runner and mesh.federation.
Separating the two is what makes the extraction possible, and it is why the package still sits
inside ember.
"""
