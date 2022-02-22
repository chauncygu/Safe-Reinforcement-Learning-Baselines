{{ name }}
{{ underline }}

.. currentmodule:: {{ module }}
.. auto{{ objtype }}:: {{ objname }}   {% if objtype == "class" %}
   :members:
   :inherited-members:
   {% endif %}
