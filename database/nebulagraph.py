import atexit
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import prettytable
from nebula3.Config import Config
from nebula3.data.DataObject import Value, ValueWrapper
from nebula3.data.ResultSet import ResultSet
from nebula3.gclient.net import ConnectionPool
from tenacity import retry, stop_after_attempt, wait_random_exponential

# from nebula3.common import *
# import json
from src.utils.base import file_exist, print_text
from src.llm.env import EmbeddingEnv

QUOTE = '"'
RETRY_TIMES = 3
WAIT_MIN_SECONDS = 0.5
WAIT_MAX_SECONDS = 10

import logging

logger = logging.getLogger(__name__)


from string import Template


# ── NebulaGraph Result Formatting ──────────────────────────────

_cast_as = {
    Value.NVAL: "as_null",
    Value.BVAL: "as_bool",
    Value.IVAL: "as_int",
    Value.FVAL: "as_double",
    Value.SVAL: "as_string",
    Value.LVAL: "as_list",
    Value.UVAL: "as_set",
    Value.MVAL: "as_map",
    Value.TVAL: "as_time",
    Value.DVAL: "as_date",
    Value.DTVAL: "as_datetime",
    Value.VVAL: "as_node",
    Value.EVAL: "as_relationship",
    Value.PVAL: "as_path",
    Value.GGVAL: "as_geography",
    Value.DUVAL: "as_duration",
}


def _cast(val: ValueWrapper):
    _type = val._value.getType()
    if _type == Value.__EMPTY__:
        return None
    if _type in _cast_as:
        return getattr(val, _cast_as[_type])()
    if _type == Value.LVAL:
        return [x.cast() for x in val.as_list()]
    if _type == Value.UVAL:
        return {x.cast() for x in val.as_set()}
    if _type == Value.MVAL:
        return {k: v.cast() for k, v in val.as_map().items()}


def result_to_df(result: ResultSet) -> pd.DataFrame:
    assert result.is_succeeded()
    columns = result.keys()
    d: Dict[str, list] = {}
    for col_num in range(result.col_size()):
        col_name = columns[col_num]
        col_list = result.column_values(col_name)
        d[col_name] = [x.cast() for x in col_list]
    return pd.DataFrame.from_dict(d, columns=columns)


def print_resp(resp: ResultSet):
    assert resp.is_succeeded(), resp
    output_table = prettytable.PrettyTable()
    output_table.field_names = resp.keys()
    for recode in resp:
        value_list = []
        for col in recode:
            val = _cast(col)
            value_list.append(val)
        output_table.add_row(value_list)
    print(output_table)


class NebulaClient:
    """Low-level NebulaGraph client wrapper managing connection pool, session, and graph space DDL."""

    def __init__(self):
        config = Config()
        config.max_connection_pool_size = 10

        self.connection_pool = ConnectionPool()
        ok = False
        try:
            ok = self.connection_pool.init([("127.0.0.1", 9669)], config)
        except Exception as e:
            logger.warning(f"Initialize ConnectionPool failed: {e}")
            ok = False

        if not ok:
            # Connection pool initialization failed, log warning and set related attributes to None to avoid AttributeError on subsequent close calls
            logger.warning(
                "Nebula ConnectionPool init failed or returned False; Nebula features will be disabled."
            )
            self.connection_pool = None
            self.session = None
            return

        # Safely catch exceptions when getting session; keep as None on failure
        try:
            self.session = self.connection_pool.get_session("root", "nebula")
        except Exception as e:
            logger.warning(f"Get session failed: {e}")
            self.session = None

        if self.session is None:
            logger.warning("Nebula session is None; nebula features may be unavailable.")
        else:
            # Only register exit hook when session is successfully obtained
            atexit.register(self._atexit_close)

    def _atexit_close(self):
        # Skip network I/O during exit phase
        if hasattr(sys, "is_finalizing") and sys.is_finalizing():
            return

        # Guard check: during interpreter exit or partial object destruction,
        # session or connection_pool may be None or already cleaned up,
        # so we need additional checks and exception handling to avoid AttributeError.
        try:
            if getattr(self, "session", None):
                try:
                    self.session.release()
                except Exception as e:
                    logger.warning(f"Close session failed: {e}")
            if getattr(self, "connection_pool", None):
                try:
                    self.connection_pool.close()
                except Exception as e:
                    logger.warning(f"Close connection pool failed: {e}")
        except Exception as e:
            logger.warning(f"_atexit_close failed: {e}")

    def create_space(self, db_name):
        """Create graph space (string VID type, for legacy triplet storage)."""
        self.session.execute(
            f"CREATE SPACE IF NOT EXISTS {db_name}(vid_type=FIXED_STRING(256), partition_num=1, replica_factor=1);"
        )
        time.sleep(10)
        self.session.execute(
            f"USE {db_name}; CREATE TAG IF NOT EXISTS entity(name string);"
        )
        self.session.execute(
            f"USE {db_name}; CREATE EDGE IF NOT EXISTS relationship(relationship string);"
        )
        self.session.execute(
            f"USE {db_name}; CREATE TAG INDEX IF NOT EXISTS entity_index ON entity(name(256));"
        )
        time.sleep(10)

    def create_graph_space(self, db_name):
        """Create graph space (INT64 VID type, for PurifyGraphRAG L2 persistent graph storage, with source_chunk attribute)."""
        self.session.execute(
            f"CREATE SPACE IF NOT EXISTS {db_name}(vid_type=INT64, partition_num=10, replica_factor=1);"
        )
        time.sleep(10)
        self.session.execute(
            f"USE {db_name}; CREATE TAG IF NOT EXISTS entity(name string, type string, source_chunk string);"
        )
        self.session.execute(
            f"USE {db_name}; CREATE EDGE IF NOT EXISTS relationship(relationship string, source_chunk string);"
        )

    def drop_space(self, db_name):
        if not isinstance(db_name, list):
            db_name = [db_name]
        for space in db_name:
            self.session.execute(f"drop space {space}")

    def info(self, db_name):
        # Switch space
        use_resp = self.session.execute(f"USE {db_name};")
        if not use_resp.is_succeeded():
            print_resp(use_resp)
            return

        submit_resp = self.session.execute("SUBMIT JOB STATS;")
        if not submit_resp.is_succeeded():
            print_resp(submit_resp)
            return

        # Wait for stats job to complete
        for _ in range(10):
            time.sleep(1)
            stats_resp = self.session.execute("SHOW STATS;")
            if stats_resp.is_succeeded():
                print_resp(stats_resp)
                return

        print("show stats still not successful, please retry later.")

    def count_edges(self, db_name):
        result = self.session.execute(
            f"use {db_name}; MATCH (m)-[e]->(n) RETURN COUNT(*);"
        )
        print_resp(result)

    def show_space(self):
        result = self.session.execute("SHOW SPACES;")
        print_resp(result)
        return result

    def get_all_db_name(self):
        # result = self.session.execute('SHOW SPACES LIKE rgb2;')
        # print('all_db_name:', result)

        result = self.session.execute("SHOW SPACES;")
        names = result.column_values(key="Name")
        # names = [x.get_value() for x in names]
        print("all_db_name:", names)
        # assert isinstance(names, list)
        return names

    def show_edges(self, db_name, limits):
        result = self.session.execute(
            f"use {db_name}; MATCH ()-[e]->() RETURN e LIMIT {limits};"
        )
        print_resp(result)

    def clear(self, db_name):
        query = f"CLEAR SPACE {db_name};"
        self.session.execute(query)

    def save_triplets(self, db_name, file_path=None):
        if not file_path:
            file_path = db_name + "_triplets.json"
        all_triples = self.get_triplets(db_name=db_name)
        from src.utils.base import save_to_json

        save_to_json(file_path=file_path, data=all_triples)

    def get_triplets(self, db_name):
        result = self.session.execute(
            f"use {db_name}; MATCH (n1)-[e]->(n2) RETURN n1, e, n2;"
        )

        all_triples = []
        print(f"db resulsts row: {result.row_size()}")

        if result.row_size() > 0:
            from tqdm import tqdm

            for row in tqdm(result.rows(), "parse db results"):
                values = row.values

                head, relation, tail = "", "", ""

                for value in values:
                    if value.field == 9:  # Corresponds to Vertex
                        vertex = value.get_vVal()
                        if not head:
                            head = vertex.vid.get_sVal().decode("utf-8")
                        else:
                            tail = vertex.vid.get_sVal().decode("utf-8")

                    elif value.field == 10:  # Corresponds to Edge
                        edge = value.get_eVal()
                        relation = (
                            edge.props.get(b"relationship").get_sVal().decode("utf-8")
                        )

                triplet = [head, relation, tail]
                all_triples.append(triplet)
        else:
            print(f"No triplets found {db_name}.")

        all_triples = set(tuple(triplet) for triplet in all_triples)
        all_triples = [list(triplet) for triplet in all_triples]

        return all_triples


rel_query_sample_edge = Template(
    """
MATCH ()-[e:`$edge_type`]->()
RETURN [src(e), dst(e)] AS sample_edge LIMIT 1
"""
)

rel_query_edge_type = Template(
    """
MATCH (m)-[:`$edge_type`]->(n)
  WHERE id(m) == $quote$src_id$quote AND id(n) == $quote$dst_id$quote
RETURN "(:" + tags(m)[0] + ")-[:$edge_type]->(:" + tags(n)[0] + ")" AS rels
"""
)


def hash_string_to_rank(string: str) -> int:
    # get signed 64-bit hash value
    signed_hash = hash(string)

    # reduce the hash value to a 64-bit range
    mask = (1 << 64) - 1
    signed_hash &= mask

    # convert the signed hash value to an unsigned 64-bit integer
    if signed_hash & (1 << 63):
        unsigned_hash = -((signed_hash ^ mask) + 1)
    else:
        unsigned_hash = signed_hash

    return unsigned_hash


def prepare_subjs_param(
    subjs: Optional[List[str]], vid_type: str = "FIXED_STRING(256)"
) -> Dict:
    """Prepare parameters for query."""
    if subjs is None:
        return {}
    from nebula3.common import ttypes

    subjs_list = []
    subjs_byte = ttypes.Value()

    # filter non-digit string for INT64 vid type
    if vid_type == "INT64":
        subjs = [subj for subj in subjs if subj.isdigit()]
        if len(subjs) == 0:
            logger.warning(
                f"KG is with INT64 vid type, but no digit string is provided."
                f"Return empty subjs, and no query will be executed."
                f"subjs: {subjs}"
            )
            return {}
    for subj in subjs:
        if not isinstance(subj, str):
            raise TypeError(f"Subject should be str, but got {type(subj).__name__}.")
        subj_byte = ttypes.Value()
        if vid_type == "INT64":
            assert subj.isdigit(), (
                "Subject should be a digit string in current "
                "graph store, where vid type is INT64."
            )
            subj_byte.set_iVal(int(subj))
        else:
            subj_byte.set_sVal(subj)
        subjs_list.append(subj_byte)
    subjs_nlist = ttypes.NList(values=subjs_list)
    subjs_byte.set_lVal(subjs_nlist)
    return {"subjs": subjs_byte}


def escape_str(value: str) -> str:
    """Escape String for NebulaGraph Query."""
    if not value:
        return ""
    patterns = {
        '"': " ",
    }
    for pattern in patterns:
        if pattern in value:
            value = value.replace(pattern, patterns[pattern])
    if value[0] == " " or value[-1] == " ":
        value = value.strip()

    return value


class NebulaDB:
    """NebulaGraph high-level graph database operation wrapper, providing CRUD for nodes/edges, Schema management, and graph query interface."""

    def __init__(
        self,
        space_name: Optional[str] = None,
        session_pool: Optional[Any] = None,
        edge_types: Optional[List[str]] = ["relationship"],
        rel_prop_names: Optional[List[str]] = ["relationship"],
        tags: Optional[List[str]] = ["entity"],
        tag_prop_names: Optional[List[str]] = ["name"],
        include_vid: bool = True,
        session_pool_kwargs: Optional[Dict[str, Any]] = {},
        server_ip="127.0.0.1",
        server_port="9669",
        **kwargs: Any,
    ) -> None:
        """Initialize NebulaGraph graph store.

        Args:
            session_pool: NebulaGraph session pool.
            space_name: NebulaGraph space name.
            edge_types: Edge types.
            rel_prop_names: Relation property names corresponding to edge types.
            tags: Tags.
            tag_prop_names: Tag property names corresponding to tags.
            session_pool_kwargs: Keyword arguments for NebulaGraph session pool.
            **kwargs: Keyword arguments.
        """

        self.server_ip = server_ip
        self.server_port = server_port

        os.environ["NEBULA_USER"] = "root"
        os.environ["NEBULA_PASSWORD"] = "nebula"  # default is "nebula"
        os.environ["NEBULA_ADDRESS"] = f"{self.server_ip}:{self.server_port}"

        self._space_name = space_name
        # self.client = NebulaClient()
        self.client = NebulaClient()
        self.entities = None

        try:
            import nebula3  # noqa
        except ImportError:
            raise ImportError(
                "Please install NebulaGraph Python client first: "
                "`pip install nebula3-python`"
            )
        assert space_name is not None, "space_name should be provided."
        self._space_name = space_name
        self._session_pool_kwargs = session_pool_kwargs

        self._session_pool: Any = session_pool
        if self._session_pool is None:
            try:
                self.init_session_pool()
            except Exception as e:
                logger.warning(f"init_session_pool failed: {e}")

        # Only register exit hook when session_pool is initialized successfully
        if getattr(self, "_session_pool", None):
            atexit.register(self._atexit_close)
        else:
            logger.warning("Session pool is not initialized; atexit handler not registered.")

        self._vid_type = self._get_vid_type()

        self._tags = tags or ["entity"]
        self._edge_types = edge_types or ["rel"]
        self._rel_prop_names = rel_prop_names or ["predicate,"]
        if len(self._edge_types) != len(self._rel_prop_names):
            raise ValueError(
                "edge_types and rel_prop_names to define relation and relation name"
                "should be provided, yet with same length."
            )
        if len(self._edge_types) == 0:
            raise ValueError("Length of `edge_types` should be greater than 0.")

        if tag_prop_names is None or len(self._tags) != len(tag_prop_names):
            raise ValueError(
                "tag_prop_names to define tag and tag property name should be "
                "provided, yet with same length."
            )

        if len(self._tags) == 0:
            raise ValueError("Length of `tags` should be greater than 0.")

        # for building query
        self._edge_dot_rel = [
            f"`{edge_type}`.`{rel_prop_name}`"
            for edge_type, rel_prop_name in zip(self._edge_types, self._rel_prop_names)
        ]

        self._edge_prop_map = {}
        for edge_type, rel_prop_name in zip(self._edge_types, self._rel_prop_names):
            self._edge_prop_map[edge_type] = [
                prop.strip() for prop in rel_prop_name.split(",")
            ]

        # cypher string like: map{`follow`: "degree", `serve`: "start_year,end_year"}
        self._edge_prop_map_cypher_string = (
            "map{"
            + ", ".join(
                [
                    f"`{edge_type}`: \"{','.join(rel_prop_names)}\""
                    for edge_type, rel_prop_names in self._edge_prop_map.items()
                ]
            )
            + "}"
        )

        # build tag_prop_names map
        self._tag_prop_names_map = {}
        for tag, prop_names in zip(self._tags, tag_prop_names or []):
            if prop_names is not None:
                self._tag_prop_names_map[tag] = f"`{tag}`.`{prop_names}`"
        self._tag_prop_names: List[str] = list(
            {
                prop_name.strip()
                for prop_names in tag_prop_names or []
                if prop_names is not None
                for prop_name in prop_names.split(",")
            }
        )

        self._include_vid = include_vid

    def init_session_pool(self) -> Any:
        """Return NebulaGraph session pool."""
        from nebula3.Config import SessionPoolConfig
        from nebula3.gclient.net.SessionPool import SessionPool

        # ensure "NEBULA_USER", "NEBULA_PASSWORD", "NEBULA_ADDRESS" are set
        # in environment variables
        if not all(
            key in os.environ
            for key in ["NEBULA_USER", "NEBULA_PASSWORD", "NEBULA_ADDRESS"]
        ):
            raise ValueError(
                "NEBULA_USER, NEBULA_PASSWORD, NEBULA_ADDRESS should be set in "
                "environment variables when NebulaGraph Session Pool is not "
                "directly passed."
            )
        graphd_host, graphd_port = os.environ["NEBULA_ADDRESS"].split(":")
        session_pool = SessionPool(
            os.environ["NEBULA_USER"],
            os.environ["NEBULA_PASSWORD"],
            self._space_name,
            [(graphd_host, int(graphd_port))],
        )

        seesion_pool_config = SessionPoolConfig()
        session_pool.init(seesion_pool_config)
        self._session_pool = session_pool

        # print('self._session_pool', self._session_pool)
        return self._session_pool

    def _get_vid_type(self) -> str:
        """Get vid type."""
        return (
            self.execute(f"DESCRIBE SPACE {self._space_name}")
            .column_values("Vid Type")[0]
            .cast()
        )

    @retry(
        wait=wait_random_exponential(min=WAIT_MIN_SECONDS, max=WAIT_MAX_SECONDS),
        stop=stop_after_attempt(RETRY_TIMES),
    )
    def execute(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        """Execute query.

        Args:
            query: Query.
            param_map: Parameter map.

        Returns:
            Query result.
        """
        from nebula3.Exception import IOErrorException
        from nebula3.fbthrift.transport.TTransport import TTransportException

        # Clean the query string by removing triple backticks
        query = query.replace("```", "").strip()

        try:
            result = self._session_pool.execute_parameter(query, param_map)
            if result is None:
                raise ValueError(f"Query failed. Query: {query}, Param: {param_map}")
            if not result.is_succeeded():
                raise ValueError(
                    f"Query failed. Query: {query}, Param: {param_map}"
                    f"Error message: {result.error_msg()}"
                )
            return result
        except (TTransportException, IOErrorException, RuntimeError) as e:
            logger.error(
                f"Connection issue, try to recreate session pool. Query: {query}, "
                f"Param: {param_map}"
                f"Error: {e}"
            )
            self.init_session_pool()
            logger.info(
                f"Session pool recreated. Query: {query}, Param: {param_map}"
                f"This was due to error: {e}, and now retrying."
            )
            raise

        except ValueError as e:
            # query failed on db side
            logger.error(
                f"Query failed. Query: {query}, Param: {param_map}"
                f"Error message: {e}"
            )
            raise
        except Exception as e:
            # other exceptions
            logger.error(
                f"Query failed. Query: {query}, Param: {param_map}"
                f"Error message: {e}"
            )
            raise

    # def __del__(self):
    #     del self.client

    def upsert_triplet(self, subj: str, rel: str, obj: str) -> None:
        """Add triplet."""
        # Note, to enable leveraging existing knowledge graph,
        # the (triplet -- property graph) mapping
        #   makes (n:1) edge_type.prop_name --> triplet.rel
        # thus we have to assume rel to be the first edge_type.prop_name
        # here in upsert_triplet().
        # This applies to the type of entity(tags) with subject and object, too,
        # thus we have to assume subj to be the first entity.tag_name

        # lower case subj, rel, obj
        subj = escape_str(subj)
        rel = escape_str(rel)
        obj = escape_str(obj)
        if self._vid_type == "INT64":
            assert all(
                [subj.isdigit(), obj.isdigit()]
            ), "Subject and object should be digit strings in current graph store."
            subj_field = subj
            obj_field = obj
        else:
            subj_field = f"{QUOTE}{subj}{QUOTE}"
            obj_field = f"{QUOTE}{obj}{QUOTE}"
        edge_field = f"{subj_field}->{obj_field}"

        edge_type = self._edge_types[0]
        rel_prop_name = self._rel_prop_names[0]
        entity_type = self._tags[0]
        rel_hash = hash_string_to_rank(rel)
        dml_query = (
            f"INSERT VERTEX `{entity_type}`(name) "
            f"  VALUES {subj_field}:({QUOTE}{subj}{QUOTE});"
            f"INSERT VERTEX `{entity_type}`(name) "
            f"  VALUES {obj_field}:({QUOTE}{obj}{QUOTE});"
            f"INSERT EDGE `{edge_type}`(`{rel_prop_name}`) "
            f"  VALUES "
            f"{edge_field}"
            f"@{rel_hash}:({QUOTE}{rel}{QUOTE});"
        )
        logger.debug(f"upsert_triplet()\nDML query: {dml_query}")
        result = self.execute(dml_query)
        assert (
            result and result.is_succeeded()
        ), f"Failed to upsert triplet: {subj} {rel} {obj}, query: {dml_query}"

    def _format_vid(self, vid: Any) -> str:
        vid = escape_str(str(vid))
        if self._vid_type == "INT64":
            assert vid.isdigit(), "Vertex id should be a digit string for INT64 space."
            return vid
        return f"{QUOTE}{vid}{QUOTE}"

    def _filter_props(self, props: Optional[Dict[str, Any]], allowed: List[str]) -> Dict[str, Any]:
        if not props:
            return {}
        return {k: v for k, v in props.items() if k in allowed and v is not None}

    def upsert_vertex(self, vertex_id: Any, properties: Optional[Dict[str, Any]] = None) -> None:
        tag = self._tags[0]
        vid_field = self._format_vid(vertex_id)
        # For example space: entity(name, type, source_chunks)
        allowed = list(set((self._tag_prop_names or []) + ["name", "type", "source_chunk"]))
        props = self._filter_props(properties, allowed)

        if props:
            prop_names = ", ".join(props.keys())
            prop_values = []
            for v in props.values():
                if isinstance(v, str):
                    prop_values.append(f"{QUOTE}{escape_str(v)}{QUOTE}")
                else:
                    prop_values.append(str(v))
            value_str = ", ".join(prop_values)
            dml_query = (
                f"INSERT VERTEX `{tag}`({prop_names}) "
                f"VALUES {vid_field}:({value_str});"
            )
        else:
            dml_query = f"INSERT VERTEX `{tag}`() VALUES {vid_field}:();"

        logger.debug(f"upsert_vertex()\nDML query: {dml_query}")
        result = self.execute(dml_query)
        assert result and result.is_succeeded(), f"Failed to upsert vertex: {vertex_id}"

    def upsert_edge(
        self,
        src_id: Any,
        tgt_id: Any,
        relation: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        edge_type = self._edge_types[0]
        rel_prop_name = self._rel_prop_names[0]
        rel = escape_str(relation)

        src_field = self._format_vid(src_id)
        tgt_field = self._format_vid(tgt_id)
        edge_field = f"{src_field}->{tgt_field}"

        # For example space: relationship(relationship, source_chunk)
        allowed = list(set(self._edge_prop_map.get(edge_type, []) + ["relationship", "source_chunk"]))
        props = self._filter_props(properties, allowed)
        if rel_prop_name and rel_prop_name not in props:
            props[rel_prop_name] = rel

        prop_names = ", ".join(props.keys())
        prop_values = []
        for v in props.values():
            if isinstance(v, str):
                prop_values.append(f"{QUOTE}{escape_str(v)}{QUOTE}")
            else:
                prop_values.append(str(v))
        value_str = ", ".join(prop_values)

        rel_hash = hash_string_to_rank(rel)
        dml_query = (
            f"INSERT EDGE `{edge_type}`({prop_names}) "
            f"VALUES {edge_field}@{rel_hash}:({value_str});"
        )
        logger.debug(f"upsert_edge()\nDML query: {dml_query}")
        result = self.execute(dml_query)
        assert result and result.is_succeeded(), (
            f"Failed to upsert edge: {src_id} {relation} {tgt_id}"
        )

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the NebulaGraph store."""
        if self.schema and not refresh:
            return self.schema
        self.refresh_schema()
        logger.debug(f"get_schema()\nschema: {self.schema}")
        return self.schema

    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet.
        1. Similar to upsert_triplet(),
           we have to assume rel to be the first edge_type.prop_name.
        2. After edge being deleted, we need to check if the subj or
           obj are isolated vertices,
           if so, delete them, too.
        """
        # lower case subj, rel, obj
        subj = escape_str(subj)
        rel = escape_str(rel)
        obj = escape_str(obj)

        if self._vid_type == "INT64":
            assert all(
                [subj.isdigit(), obj.isdigit()]
            ), "Subject and object should be digit strings in current graph store."
            subj_field = subj
            obj_field = obj
        else:
            subj_field = f"{QUOTE}{subj}{QUOTE}"
            obj_field = f"{QUOTE}{obj}{QUOTE}"
        edge_field = f"{subj_field}->{obj_field}"

        # DELETE EDGE serve "player100" -> "team204"@7696463696635583936;
        edge_type = self._edge_types[0]
        # rel_prop_name = self._rel_prop_names[0]
        rel_hash = hash_string_to_rank(rel)
        dml_query = f"DELETE EDGE `{edge_type}`" f"  {edge_field}@{rel_hash};"
        logger.debug(f"delete()\nDML query: {dml_query}")
        result = self.execute(dml_query)
        assert (
            result and result.is_succeeded()
        ), f"Failed to delete triplet: {subj} {rel} {obj}, query: {dml_query}"
        # Get isolated vertices to be deleted
        # MATCH (s) WHERE id(s) IN ["player700"] AND NOT (s)-[]-()
        # RETURN id(s) AS isolated
        query = (
            f"MATCH (s) "
            f"  WHERE id(s) IN [{subj_field}, {obj_field}] "
            f"  AND NOT (s)-[]-() "
            f"RETURN id(s) AS isolated"
        )
        result = self.execute(query)
        isolated = result.column_values("isolated")
        if not isolated:
            return
        # DELETE VERTEX "player700" or DELETE VERTEX 700
        quote_field = QUOTE if self._vid_type != "INT64" else ""
        vertex_ids = ",".join(
            [f"{quote_field}{v.cast()}{quote_field}" for v in isolated]
        )
        dml_query = f"DELETE VERTEX {vertex_ids};"

        result = self.execute(dml_query)
        assert (
            result and result.is_succeeded()
        ), f"Failed to delete isolated vertices: {isolated}, query: {dml_query}"

    def refresh_schema(self) -> None:
        """
        Refreshes the NebulaGraph Store Schema.
        """
        tags_schema, edge_types_schema, relationships = [], [], []
        for tag in self.execute("SHOW TAGS").column_values("Name"):
            tag_name = tag.cast()
            tag_schema = {"tag": tag_name, "properties": []}
            r = self.execute(f"DESCRIBE TAG `{tag_name}`")
            props, types, comments = (
                r.column_values("Field"),
                r.column_values("Type"),
                r.column_values("Comment"),
            )
            for i in range(r.row_size()):
                # back compatible with old version of nebula-python
                property_defination = (
                    (props[i].cast(), types[i].cast())
                    if comments[i].is_empty()
                    else (props[i].cast(), types[i].cast(), comments[i].cast())
                )
                tag_schema["properties"].append(property_defination)
            tags_schema.append(tag_schema)
        for edge_type in self.execute("SHOW EDGES").column_values("Name"):
            edge_type_name = edge_type.cast()
            edge_schema = {"edge": edge_type_name, "properties": []}
            r = self.execute(f"DESCRIBE EDGE `{edge_type_name}`")
            props, types, comments = (
                r.column_values("Field"),
                r.column_values("Type"),
                r.column_values("Comment"),
            )
            for i in range(r.row_size()):
                # back compatible with old version of nebula-python
                property_defination = (
                    (props[i].cast(), types[i].cast())
                    if comments[i].is_empty()
                    else (props[i].cast(), types[i].cast(), comments[i].cast())
                )
                edge_schema["properties"].append(property_defination)
            edge_types_schema.append(edge_schema)

            # build relationships types
            sample_edge = self.execute(
                rel_query_sample_edge.substitute(edge_type=edge_type_name)
            ).column_values("sample_edge")
            if len(sample_edge) == 0:
                continue
            src_id, dst_id = sample_edge[0].cast()
            r = self.execute(
                rel_query_edge_type.substitute(
                    edge_type=edge_type_name,
                    src_id=src_id,
                    dst_id=dst_id,
                    quote="" if self._vid_type == "INT64" else QUOTE,
                )
            ).column_values("rels")
            if len(r) > 0:
                relationships.append(r[0].cast())

        self.schema = (
            f"Node properties: {tags_schema}\n"
            f"Edge properties: {edge_types_schema}\n"
            f"Relationships: {relationships}\n"
        )

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Dict[str, list]:
        """Execute an nGQL query and return results as a {column_name: [value_list]} dictionary."""
        result = self.execute(query, param_map)
        columns = result.keys()
        d: Dict[str, list] = {}
        for col_num in range(result.col_size()):
            col_name = columns[col_num]
            col_list = result.column_values(col_name)
            d[col_name] = [x.cast() for x in col_list]
        return d

    def get_db_name(self):
        return self._space_name

    def get_flat_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get flat rel map."""
        # The flat means for multi-hop relation path, we could get
        # knowledge like: subj -rel-> obj -rel-> obj <-rel- obj.
        # This type of knowledge is useful for some tasks.
        # +---------------------+---------------------------------------------...-----+
        # | subj                | flattened_rels                              ...     |
        # +---------------------+---------------------------------------------...-----+
        # | "{name:Tony Parker}"| "{name: Tony Parker}-[follow:{degree:95}]-> ...ili}"|
        # | "{name:Tony Parker}"| "{name: Tony Parker}-[follow:{degree:95}]-> ...r}"  |
        # ...
        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map

        # WITH map{`true`: "-[", `false`: "<-["} AS arrow_l,
        #      map{`true`: "]->", `false`: "]-"} AS arrow_r,
        #      map{`follow`: "degree", `serve`: "start_year,end_year"} AS edge_type_map
        # MATCH p=(start)-[e:follow|serve*..2]-()
        #     WHERE id(start) IN ["player100", "player101"]
        #   WITH start, id(start) AS vid, nodes(p) AS nodes, e AS rels,
        #     length(p) AS rel_count, arrow_l, arrow_r, edge_type_map
        #   WITH
        #     REDUCE(s = vid + '{', key IN [key_ in ["name"]
        #       WHERE properties(start)[key_] IS NOT NULL]  | s + key + ': ' +
        #         COALESCE(TOSTRING(properties(start)[key]), 'null') + ', ')
        #         + '}'
        #       AS subj,
        #     [item in [i IN RANGE(0, rel_count - 1) | [nodes[i], nodes[i + 1],
        #         rels[i], typeid(rels[i]) > 0, type(rels[i]) ]] | [
        #      arrow_l[tostring(item[3])] +
        #          item[4] + ':' +
        #          REDUCE(s = '{', key IN SPLIT(edge_type_map[item[4]], ',') |
        #            s + key + ': ' + COALESCE(TOSTRING(properties(item[2])[key]),
        #            'null') + ', ') + '}'
        #           +
        #      arrow_r[tostring(item[3])],
        #      REDUCE(s = id(item[1]) + '{', key IN [key_ in ["name"]
        #           WHERE properties(item[1])[key_] IS NOT NULL]  | s + key + ': ' +
        #           COALESCE(TOSTRING(properties(item[1])[key]), 'null') + ', ') + '}'
        #      ]
        #   ] AS rels
        #   WITH
        #       REPLACE(subj, ', }', '}') AS subj,
        #       REDUCE(acc = collect(NULL), l in rels | acc + l) AS flattened_rels
        #   RETURN
        #     subj,
        #     REPLACE(REDUCE(acc = subj,l in flattened_rels|acc + ' ' + l),
        #       ', }', '}')
        #       AS flattened_rels
        #   LIMIT 30

        # Based on self._include_vid
        # {name: Tim Duncan} or player100{name: Tim Duncan} for entity
        s_prefix = "vid + '{'" if self._include_vid else "'{'"
        s1 = "id(item[1]) + '{'" if self._include_vid else "'{'"

        query = (
            f"WITH map{{`true`: '-[', `false`: '<-['}} AS arrow_l,"
            f"     map{{`true`: ']->', `false`: ']-'}} AS arrow_r,"
            f"     {self._edge_prop_map_cypher_string} AS edge_type_map "
            f"MATCH p=(start)-[e:`{'`|`'.join(self._edge_types)}`*..{depth}]-() "
            f"  WHERE id(start) IN $subjs "
            f"WITH start, id(start) AS vid, nodes(p) AS nodes, e AS rels,"
            f"  length(p) AS rel_count, arrow_l, arrow_r, edge_type_map "
            f"WITH "
            f"  REDUCE(s = {s_prefix}, key IN [key_ in {self._tag_prop_names!s} "
            f"    WHERE properties(start)[key_] IS NOT NULL]  | s + key + ': ' + "
            f"      COALESCE(TOSTRING(properties(start)[key]), 'null') + ', ')"
            f"      + '}}'"
            f"    AS subj,"
            f"  [item in [i IN RANGE(0, rel_count - 1)|[nodes[i], nodes[i + 1],"
            f"      rels[i], typeid(rels[i]) > 0, type(rels[i]) ]] | ["
            f"    arrow_l[tostring(item[3])] +"
            f"      item[4] + ':' +"
            f"      REDUCE(s = '{{', key IN SPLIT(edge_type_map[item[4]], ',') | "
            f"        s + key + ': ' + COALESCE(TOSTRING(properties(item[2])[key]),"
            f"        'null') + ', ') + '}}'"
            f"      +"
            f"    arrow_r[tostring(item[3])],"
            f"    REDUCE(s = {s1}, key IN [key_ in "
            f"        {self._tag_prop_names!s} WHERE properties(item[1])[key_] "
            f"        IS NOT NULL]  | s + key + ': ' + "
            f"        COALESCE(TOSTRING(properties(item[1])[key]), 'null') + ', ')"
            f"        + '}}'"
            f"    ]"
            f"  ] AS rels "
            f"WITH "
            f"  REPLACE(subj, ', }}', '}}') AS subj,"
            f"  REDUCE(acc = collect(NULL), l in rels | acc + l) AS flattened_rels "
            f"RETURN "
            f"  subj,"
            f"  REPLACE(REDUCE(acc = subj, l in flattened_rels | acc + ' ' + l), "
            f"    ', }}', '}}') "
            f"    AS flattened_rels"
            f"  LIMIT {limit}"
        )
        subjs_param = prepare_subjs_param(subjs, self._vid_type)
        logger.debug(f"get_flat_rel_map()\nsubjs_param: {subjs},\nquery: {query}")
        if subjs_param == {}:
            # This happens when subjs is None after prepare_subjs_param()
            # Probably because vid type is INT64, but no digit string is provided.
            return rel_map
        result = self.execute(query, subjs_param)
        if result is None:
            return rel_map

        # get raw data
        subjs_ = result.column_values("subj") or []
        rels_ = result.column_values("flattened_rels") or []

        for subj, rel in zip(subjs_, rels_):
            subj_ = subj.cast()
            rel_ = rel.cast()
            if subj_ not in rel_map:
                rel_map[subj_] = []
            rel_map[subj_].append(rel_)
        return rel_map

    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get rel map."""
        # We put rels in a long list for depth>= 1, this is different from
        # SimpleGraphStore.get_rel_map() though.
        # But this makes more sense for multi-hop relation path.

        if subjs is not None:
            subjs = [
                escape_str(subj) for subj in subjs if isinstance(subj, str) and subj
            ]
            if len(subjs) == 0:
                return {}

        return self.get_flat_rel_map(subjs, depth, limit)

    def get_knowledge_sequence(self, rel_map):
        knowledge_sequence = []
        if rel_map:
            knowledge_sequence.extend(
                [str(rel_obj) for rel_objs in rel_map.values() for rel_obj in rel_objs]
            )
        else:
            print("> No knowledge sequence extracted from entities.")
            return []
        return knowledge_sequence

    def clean_sequence(
        self,
        sequence,
        name_pattern=r"(?<=\{name: )([^{}]+)(?=\})",
        edge_pattern=r"(?<=\{relationship: )([^{}]+)(?=\})",
    ):
        """
        kg result: 'James{name: James} -[relationship:{relationship: Joined}]-> Michael jordan{name: Michael jordan}'

        clean the kg result above to James -Joined-> Michael jordan
        """
        names = re.findall(name_pattern, sequence)
        edges = re.findall(edge_pattern, sequence)
        assert len(names) == sequence.count("{name:"), sequence
        assert len(edges) == sequence.count("{relationship:")
        for name in names:
            sequence = sequence.replace(f"{{name: {name}}}", "")
        for edge in edges:
            sequence = sequence.replace(
                f"[relationship:{{relationship: {edge}}}]", f"{edge}"
            )
        return sequence

    def clean_kg_sequences(self, knowledge_sequence):
        exit(0)  # remove this function, any dependency?
        # clean_knowledge_sequence = [
        #     self.clean_sequence(seq) for seq in knowledge_sequence
        # ]
        # return clean_knowledge_sequence

    def clean_rel_map(self, rel_map):
        name_pattern = r"(?<=\{name: )([^{}]+)(?=\})"
        clean_rel_map = {}
        for entity, sequences in rel_map.items():
            name = re.findall(name_pattern, entity)[0]
            clean_ent = entity.replace(f"{{name: {name}}}", "")
            clean_seq = [self.clean_sequence(seq) for seq in sequences]
            clean_rel_map[clean_ent] = clean_seq
        return clean_rel_map

    def _atexit_close(self):
        """Fallback close before process exit, no network I/O to avoid NoneType errors."""
        # Do not perform any I/O when the interpreter is finalizing
        if hasattr(sys, "is_finalizing") and sys.is_finalizing():
            return

        self._session_pool.close()

    def drop(self):
        self.client.drop_space(self._space_name)

    def info(self):
        self.client.info(self._space_name)

    def count_edges(self):
        self.client.count_edges(self._space_name)

    def show_edges(self, limits=10):
        self.client.show_edges(self._space_name, limits)

    def clear(self):
        self.client.clear(self._space_name)

    def show_space(self):
        return self.client.show_space()

    def get_triplets(self):
        return self.client.get_triplets(self._space_name)

    def save_triplets(self, file_path=None):
        self.client.save_triplets(self._space_name, file_path)

    def get_all_entities(self):
        from utils import file_exist, read_json, save_to_json

        # base_dir = os.path.expanduser("~")
        # triplet_dir = os.path.join(base_dir, "DepCache/database/triplets")
        triplet_dir = "/home/shuyurui/DepCache/filtered_triplets"
        entities_file = os.path.join(triplet_dir, f"{self._space_name}_entities.json")

        if file_exist(entities_file):
            print(f"load entities from {entities_file}")
            entities = read_json(entities_file)
        else:
            triplets_file = os.path.join(
                triplet_dir, f"{self._space_name}_triplets.json"
            )

            if file_exist(triplets_file):
                all_triplets = read_json(triplets_file)
            else:
                all_triplets = self.get_triplets()
            left_entities = [triplet[0] for triplet in all_triplets]
            right_entities = [triplet[2] for triplet in all_triplets]
            entities = sorted(list(set(left_entities + right_entities)))

            # assert len(entities - entities1) == 0 and len(entities1 - entities) == 0
            save_to_json(entities_file, entities)

            # print(f'triplets: {len(all_triplets)}, entities: {len(entities)}')
        print(f"entities: {len(entities)}")

        return set(entities)

    def generate_embedding(self):
        file_path = (
            f"/home/hdd/dataset/rag-data/{self._space_name}-triplet-embedding.npz"
        )

        if file_exist(file_path):
            print(f"load embedding from {file_path}")
            loaded_data = np.load(file_path, allow_pickle=True)
            triplet2id = loaded_data["triplet2id"].item()
            triplet_embeddings = loaded_data["triplet_embeddings"]
            print(f"load {len(triplet_embeddings)} embeddings")

            return triplet2id, triplet_embeddings

        all_triplets = self.get_triplets()
        triplet2id = {}
        all_triplets_str = []
        for i, triplet in enumerate(all_triplets):
            triplet_str = " ".join(triplet)
            triplet2id[triplet_str] = i
            all_triplets_str.append(triplet_str)

        embed_model = EmbeddingEnv(
            embed_name="BAAI/bge-small-en-v1.5", embed_batch_size=10
        )

        all_embeddings = []

        step = 400
        n_triplets = len(all_triplets_str)
        for start in range(0, n_triplets, step):
            input_texts = all_triplets_str[start : min(start + step, n_triplets)]
            # print(input_texts)
            embeddings = embed_model.get_embeddings(input_texts)
            all_embeddings += embeddings
            # break

        # for i, triplet in enumerate(all_triplets_str):
        #     print(i, triplet)
        #     embedding = embed_model.get_embedding(triplet)
        #     assert np.allclose(embedding, all_embeddings[i], atol=1e-4), i

        all_embeddings_np = np.array(all_embeddings, dtype=float)

        np.savez(file_path, triplet2id=triplet2id, triplet_embeddings=all_embeddings_np)

        print(f"triplet embeddings ({all_embeddings_np.shape}) saved to {file_path}")
        return triplet2id, all_embeddings_np

    def load_triplets_embedding(self, file_path):

        self.client.save_triplets(self._space_name, file_path)

    def two_hop_parse_triplets(self, query):
        # Define regex patterns
        two_hop_pattern1 = re.compile(
            r"(.+) <-(?<! )(.+?)(?<! )- (.+) -(?<! )(.+?)(?<! )-> (.+)"
        )
        two_hop_pattern2 = re.compile(
            r"(.+) <-(?<! )(.+?)(?<! )- (.+) <-(?<! )(.+?)(?<! )- (.+)"
        )
        two_hop_pattern3 = re.compile(
            r"(.+) -(?<! )(.+?)(?<! )-> (.+) -(?<! )(.+?)(?<! )-> (.+)"
        )
        two_hop_pattern4 = re.compile(
            r"(.+) -(?<! )(.+?)(?<! )-> (.+) <-(?<! )(.+?)(?<! )- (.+)"
        )

        one_hop_pattern5 = re.compile(r"(.+) -(?<! )(.+?)(?<! )-> (.+)")
        one_hop_pattern6 = re.compile(r"(.+) <-(?<! )(.+?)(?<! )- (.+)")

        match = two_hop_pattern1.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity2, relation1, entity1), (entity2, relation2, entity3)]

        match = two_hop_pattern2.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity2, relation1, entity1), (entity3, relation2, entity2)]

        match = two_hop_pattern3.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity1, relation1, entity2), (entity2, relation2, entity3)]

        match = two_hop_pattern4.match(query)
        if match:
            entity1, relation1, entity2, relation2, entity3 = match.groups()
            return [(entity1, relation1, entity2), (entity3, relation2, entity2)]

        match = one_hop_pattern5.match(query)
        if match:
            entity1, relation1, entity2 = match.groups()
            return [(entity1, relation1, entity2)]

        match = one_hop_pattern6.match(query)
        if match:
            entity1, relation1, entity2 = match.groups()
            return [(entity2, relation1, entity1)]

        assert False, query

    def rel_map_to_triplets(self, clean_map):
        all_triplets = set()
        for rels in clean_map.values():
            triplets, _ = self.two_hop_parse_multi_triplets(rels)
            all_triplets.update(triplets)
        return all_triplets

    def kg_seqs_to_triplets(self, kg_seqs):
        time_parse_triplet = -time.time()
        all_triplets = []
        for rel in kg_seqs:
            for triplet in self.two_hop_parse_triplets(rel):
                all_triplets.append(triplet)
        all_triplets = set(all_triplets)
        time_parse_triplet += time.time()
        print(f"parse triplets {time_parse_triplet}")
        # assert set(all_triplets) == set(all_triplets1)

        return all_triplets

    def kg_seqs_to_triplet_lists(self, kg_seqs):
        if not isinstance(kg_seqs, list):
            kg_seqs = [kg_seqs]
        return [self.two_hop_parse_triplets(rel) for rel in kg_seqs]

    def two_hop_parse_multi_triplets(self, queries):
        triplets = []
        rel_to_entities = {}
        for query in queries:
            query_triplets = self.two_hop_parse_triplets(query)
            triplets += query_triplets
            if query not in rel_to_entities:
                rel_to_entities[query] = set()
            for triplet in query_triplets:
                rel_to_entities[query].add(triplet[0])
                rel_to_entities[query].add(triplet[2])
        return triplets, rel_to_entities


if __name__ == "__main__":

    db_name = "rgb_en"
    client = NebulaClient()
    client.show_space()
    client.info(db_name)

    db_name = "rgb_en"
    db = NebulaDB(space_name=db_name)
    rel_map = db.get_rel_map(["Zhang yimou"], depth=1, limit=30)
    clean_rel_map = db.clean_rel_map(rel_map)
    knowledge_sequence = db.get_knowledge_sequence(clean_rel_map)
    for i, kg in enumerate(knowledge_sequence):
        print(i, kg)

    # print(rel_map)
    # # create space
    # client = NebulaClient()
    # client.create_space('crag_small')
    # client.show_space()
    # exit(0)

    # # drop space
    # client.drop_space('rgb')
    # client.clear('rgb')

    # cd nebula-graph-studio-3.7.0
    # Run the studio web service in the background to connect to the NebulaGraph database
    # docker compose up -d
    # 2. Studio - SSH tunnel (since it's bound to localhost)
    # Run in the local terminal:
    # ssh -L 7001:127.0.0.1:7001 shuyurui@121.48.164.166 -N
    # Then open http://localhost:7001 in a browser. When connecting to Nebula in Studio, use 127.0.0.1:9669, user root, password nebula.